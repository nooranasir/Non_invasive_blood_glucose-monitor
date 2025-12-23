/************************************************************
   ESP32 + MAX30102 + BME280 + MPU6050
   Output ONLY: NIR Value + Temperature
   (All sensors remain initialized)
************************************************************/

#include <Wire.h>
#include <Adafruit_BME280.h>
#include <MAX30105.h>
#include <MPU6050.h>
#include <Preferences.h>
#include <stdio.h>
#include <WiFi.h>
#include <HTTPClient.h>

/* ---------------- SENSORS ---------------- */
Adafruit_BME280 bme;
MPU6050 mpu;
MAX30105 particleSensor;
bool bme_present = false;

/* ---------------- MODEL COEFFICIENTS ---------------- */
// Ridge Polynomial Degree-2 Model (ridge_poly2) - trained on all features
// Features: AGE, AFTER_FOOD, IS_DIABETIC, NIR_VALUE
// Polynomial features: 14 total (includes interactions and squares)
// These are initial coefficients. They are stored in Preferences and can be updated over Serial at runtime.

// Polynomial feature coefficients (14 total):
// [0] AGE, [1] AFTER_FOOD, [2] IS_DIABETIC, [3] NIR_VALUE, 
// [4] AGE^2, [5] AGE*AFTER_FOOD, [6] AGE*IS_DIABETIC, [7] AGE*NIR_VALUE,
// [8] AFTER_FOOD^2, [9] AFTER_FOOD*IS_DIABETIC, [10] AFTER_FOOD*NIR_VALUE,
// [11] IS_DIABETIC^2, [12] IS_DIABETIC*NIR_VALUE, [13] NIR_VALUE^2
float COEF_POLY[14] = {
  -0.36225135429665917,
  0.3648857993254296,
  -0.048919652245716566,
  13.596468513258403,
  -12.590466390785922,
  -0.5294807872357575,
  0.025663418164886487,
  7.886699183225784,
  -1.2612108654482315,
  -0.10435453816108813,
  1.6425253542136424,
  -0.18344869592407181,
  -5.711123065757422,
  0.3648900717290253
};

float INTERCEPT_C = 124.92864643991247;
float MODEL_R2 = 0.5292571762262603;
float MODEL_RMSE = 15.080676691509607;

// For normalization (if needed for other models)
float SCALE = 100000.0; // normalize rawIR (1e5)

// Legacy single-feature polynomial coefficients (a*x^2 + b*x + c) on x = NIR/SCALE
float COEF_A = -96.81760438456419;
float COEF_B = 567.2537170667338;
float COEF_C = -675.2034783624928;
float SCALE_NIR = 100000.0;
bool use_single_model = false;

Preferences prefs;
String serialBuffer = "";
// --- Network / ThingSpeak configuration ---
String wifi_ssid = ""; // set over Serial: {"ssid":"..."}
String wifi_pass = "";
String thingspeak_key = ""; // set over Serial: {"ts_key":"..."}
bool wifi_connected = true;
unsigned long lastThingSpeakMillis = 0;
const unsigned long THINGSPEAK_INTERVAL_MS = 20000UL; // ThingSpeak rate limit: ~15s, use 20s
unsigned long lastWifiTryMillis = 0;
const unsigned long WIFI_RETRY_INTERVAL_MS = 30000UL;

// Helper: parse a string value for a key from a simple JSON-like string
String parseJsonString(const String &s, const char *key) {
  int ki = s.indexOf(String(key));
  if (ki < 0) return String("");
  int col = s.indexOf(':', ki);
  if (col < 0) return String("");
  int i = col + 1;
  // skip spaces
  while (i < s.length() && isSpace(s[i])) i++;
  // skip optional quotes
  if (i < s.length() && (s[i] == '"' || s[i] == '\'')) {
    char quote = s[i];
    i++;
    int start = i;
    while (i < s.length() && s[i] != quote) i++;
    if (i <= start) return String("");
    return s.substring(start, i);
  } else {
    // unquoted token until comma or closing brace
    int start = i;
    while (i < s.length()) {
      char c = s[i];
      if (c == ',' || c == '}' || c == ']') break;
      i++;
    }
    {
      String tmp = s.substring(start, i);
      tmp.trim();
      return tmp;
    }
  }
}

// Connect to WiFi using wifi_ssid and wifi_pass. Returns true if connected.
bool connectWiFi(unsigned long timeoutMs = 15000) {
  if (wifi_ssid.length() == 0) return false;
  Serial.print("Connecting to WiFi '"); Serial.print(wifi_ssid); Serial.println("'...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(wifi_ssid.c_str(), wifi_pass.c_str());
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeoutMs) {
    delay(250);
    Serial.print('.');
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifi_connected = true;
    Serial.println();
    Serial.print("WiFi connected, IP: "); Serial.println(WiFi.localIP());
    return true;
  } else {
    wifi_connected = false;
    Serial.println();
    Serial.println("WiFi connect failed or timed out");
    return false;
  }
}

// Send a single ThingSpeak update for glucose, temp, and rawIR
void sendToThingSpeak(float glucose, float tempC, uint32_t rawIR) {
  if (thingspeak_key.length() == 0) {
    Serial.println("ThingSpeak key not set; skipping upload.");
    return;
  }
  if (!wifi_connected) {
    // try to reconnect once
    if (!connectWiFi(8000)) {
      Serial.println("Not connected to WiFi; cannot send to ThingSpeak.");
      return;
    }
  }

  HTTPClient http;
  String url = "http://api.thingspeak.com/update?api_key=" + thingspeak_key;
  // field1 = glucose, field2 = temperature, field3 = rawIR
  url += "&field1=" + String(glucose, 2);
  url += "&field2=" + String(tempC, 2);
  url += "&field3=" + String(rawIR);

  Serial.print("ThingSpeak URL: "); Serial.println(url);
  http.begin(url);
  int httpCode = http.GET();
  if (httpCode > 0) {
    Serial.print("ThingSpeak HTTP response code: "); Serial.println(httpCode);
    String payload = http.getString();
    Serial.print("ThingSpeak response: "); Serial.println(payload);
  } else {
    Serial.print("ThingSpeak HTTP request failed, error: "); Serial.println(httpCode);
  }
  http.end();
}
// Scaler parameters for StandardScaler (mean and scale per feature)
float SCALER_MEAN[4] = {0,0,0,0};
float SCALER_SCALE[4] = {1,1,1,1};

// Helper: parse a float value for a key from a simple JSON-like string
float parseJsonFloat(const String &s, const char *key, float defaultVal) {
  int ki = s.indexOf(String(key));
  if (ki < 0) return defaultVal;
  int col = s.indexOf(':', ki);
  if (col < 0) return defaultVal;
  int i = col + 1;
  // skip spaces
  while (i < s.length() && isSpace(s[i])) i++;
  int start = i;
  // accept sign, digits, decimal point, exponent
  while (i < s.length()) {
    char c = s[i];
    if ((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E') {
      i++; continue;
    }
    break;
  }
  String num = s.substring(start, i);
  if (num.length() == 0) return defaultVal;
  return num.toFloat();
}

// Parse a JSON array of floats like "coef_poly":[1.0, 2.0, 3.0]
bool parseJsonFloatArray(const String &s, const char *key, float *arr, int arrLen) {
  int ki = s.indexOf(String(key));
  if (ki < 0) return false;
  int start_bracket = s.indexOf('[', ki);
  if (start_bracket < 0) return false;
  int end_bracket = s.indexOf(']', start_bracket);
  if (end_bracket < 0) return false;
  
  String arrStr = s.substring(start_bracket + 1, end_bracket);
  int count = 0;
  int pos = 0;
  while (pos < arrStr.length() && count < arrLen) {
    int nextComma = arrStr.indexOf(',', pos);
    if (nextComma < 0) nextComma = arrStr.length();
    
    String numStr = arrStr.substring(pos, nextComma);
    numStr.trim();
    arr[count++] = numStr.toFloat();
    pos = nextComma + 1;
  }
  return count == arrLen;
}

// Process an incoming JSON-like serial line and update coefficients
void processSerialJson(const String &line) {
  // Always check for network configuration keys and persist them if present
  String new_ssid = parseJsonString(line, "ssid");
  String new_pass = parseJsonString(line, "pass");
  String new_ts_key = parseJsonString(line, "ts_key");
  if (new_ssid.length() > 0 || new_pass.length() > 0 || new_ts_key.length() > 0) {
    // update runtime values
    if (new_ssid.length() > 0) wifi_ssid = new_ssid;
    if (new_pass.length() > 0) wifi_pass = new_pass;
    if (new_ts_key.length() > 0) thingspeak_key = new_ts_key;
    // persist into NVS under namespace 'net'
    prefs.begin("net", false);
    prefs.putString("ssid", wifi_ssid);
    prefs.putString("pass", wifi_pass);
    prefs.putString("ts_key", thingspeak_key);
    prefs.end();
    Serial.println("Saved network/ThingSpeak configuration to preferences.");
    // attempt immediate connect if we have SSID
    if (wifi_ssid.length() > 0) connectWiFi(8000);
  }

  // Try to parse polynomial coefficients
  float tempCoef[14];
  float tempIntercept = parseJsonFloat(line, "intercept_c", INTERCEPT_C);
  float tempR2 = parseJsonFloat(line, "r2", MODEL_R2);
  float tempRMSE = parseJsonFloat(line, "rmse", MODEL_RMSE);
  
  if (parseJsonFloatArray(line, "coef_poly", tempCoef, 14)) {
    // Successfully parsed array
    memcpy(COEF_POLY, tempCoef, sizeof(tempCoef));
    INTERCEPT_C = tempIntercept;
    MODEL_R2 = tempR2;
    MODEL_RMSE = tempRMSE;
    
    // persist
    char key[8];
    for (int i = 0; i < 14; i++) {
      snprintf(key, sizeof(key), "c%d", i);
      prefs.putFloat(key, COEF_POLY[i]);
    }
    prefs.putFloat("ic", INTERCEPT_C);
    prefs.putFloat("r2", MODEL_R2);
    prefs.putFloat("rmse", MODEL_RMSE);

    // parse and persist scaler params if present
    float tmpMean[4];
    float tmpScale[4];
    if (parseJsonFloatArray(line, "scaler_mean", tmpMean, 4) && parseJsonFloatArray(line, "scaler_scale", tmpScale, 4)) {
      for (int i = 0; i < 4; i++) {
        SCALER_MEAN[i] = tmpMean[i];
        SCALER_SCALE[i] = tmpScale[i];
        snprintf(key, sizeof(key), "sm%d", i);
        prefs.putFloat(key, SCALER_MEAN[i]);
        snprintf(key, sizeof(key), "ss%d", i);
        prefs.putFloat(key, SCALER_SCALE[i]);
      }
    }
    
    Serial.println("Model coefficients (polynomial) updated and saved:");
    for (int i = 0; i < 14; i++) {
      Serial.print("coef["); Serial.print(i); Serial.print("]=");
      Serial.println(COEF_POLY[i]);
    }
    Serial.print("intercept_c="); Serial.println(INTERCEPT_C);
    Serial.print("r2="); Serial.println(MODEL_R2);
    Serial.print("rmse="); Serial.println(MODEL_RMSE);
  } else {
    // Try legacy single-feature fields: coef_a, coef_b, intercept_c, scale
    int ki_a = line.indexOf("coef_a");
    int ki_b = line.indexOf("coef_b");
    int ki_c = line.indexOf("intercept_c");
    if (ki_a >= 0 && ki_b >= 0 && ki_c >= 0) {
      float ta = parseJsonFloat(line, "coef_a", COEF_A);
      float tb = parseJsonFloat(line, "coef_b", COEF_B);
      float tc = parseJsonFloat(line, "intercept_c", COEF_C);
      float ts = parseJsonFloat(line, "scale", SCALE_NIR);

      COEF_A = ta; COEF_B = tb; COEF_C = tc; SCALE_NIR = ts;
      use_single_model = true;

      prefs.putFloat("a", COEF_A);
      prefs.putFloat("b", COEF_B);
      prefs.putFloat("c", COEF_C);
      prefs.putFloat("s", SCALE_NIR);

      Serial.println("Legacy single-feature model updated and saved:");
      Serial.print("coef_a="); Serial.println(COEF_A);
      Serial.print("coef_b="); Serial.println(COEF_B);
      Serial.print("intercept_c="); Serial.println(COEF_C);
      Serial.print("scale="); Serial.println(SCALE_NIR);
    } else {
      Serial.println("Failed to parse coefficient array from JSON");
    }
  }
}

void checkSerialCommands() {
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (serialBuffer.length() > 0) {
        String line = serialBuffer;
        line.trim();
        if (line.startsWith("{")) {
          processSerialJson(line);
        } else if (line.startsWith("GET")) {
          Serial.println("\n========== CURRENT MODEL ==========");
            if (use_single_model) {
              Serial.print("Model Type: single-feature polynomial (NIR)\n");
              Serial.print("coef_a="); Serial.println(COEF_A);
              Serial.print("coef_b="); Serial.println(COEF_B);
              Serial.print("intercept_c="); Serial.println(COEF_C);
              Serial.print("scale="); Serial.println(SCALE_NIR);
              Serial.print("Model R²: "); Serial.println(MODEL_R2);
              Serial.print("Model RMSE: "); Serial.println(MODEL_RMSE);
            } else {
              Serial.print("Model Type: ridge_poly2 (14-feature polynomial)\n");
              Serial.print("R² Score: "); Serial.println(MODEL_R2);
              Serial.print("RMSE: "); Serial.println(MODEL_RMSE);
              Serial.println("Polynomial Coefficients:");
              for (int i = 0; i < 14; i++) {
                Serial.print("  ["); Serial.print(i); Serial.print("] = ");
                Serial.println(COEF_POLY[i]);
              }
              Serial.print("Intercept: "); Serial.println(INTERCEPT_C);
            }
            // Print network status
            Serial.println("\n--- Network / ThingSpeak ---");
            Serial.print("WiFi SSID: "); Serial.println(wifi_ssid.length() ? wifi_ssid : String("(not set)"));
            Serial.print("WiFi connected: "); Serial.println(wifi_connected ? "yes" : "no");
            Serial.print("ThingSpeak Key set: "); Serial.println(thingspeak_key.length() ? "yes" : "no");
        } else {
          Serial.println("Unrecognized command. Send JSON with model coefficients or 'GET' to print current model.");
        }
        serialBuffer = "";
      }
    } else {
      serialBuffer += ch;
      // guard length
      if (serialBuffer.length() > 2048) serialBuffer = serialBuffer.substring(serialBuffer.length()-2048);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  Serial.println("Initializing Sensors...");

  /* ----------- BME280 ----------- */
  // Try common addresses 0x76 then 0x77. If both fail, print I2C scan for debugging.
  if (bme.begin(0x76)) {
    Serial.println("BME280 OK (0x76)");
    bme_present = true;
  } else if (bme.begin(0x77)) {
    Serial.println("BME280 OK (0x77)");
    bme_present = true;
  } else {
    Serial.println("BME280 FAIL (no response at 0x76/0x77)");
    bme_present = false;
    // provide i2c scan to help diagnose wiring/address issues
    Serial.println("Performing I2C scan to detect devices on the bus...");
    for (uint8_t addr = 1; addr < 127; addr++) {
      Wire.beginTransmission(addr);
      uint8_t err = Wire.endTransmission();
      if (err == 0) {
        Serial.print(" I2C device found at 0x");
        if (addr < 16) Serial.print('0');
        Serial.println(addr, HEX);
      }
      delay(2);
    }
  }

  /* ----------- MPU6050 ----------- */
  mpu.initialize();
  Serial.println(mpu.testConnection() ? "MPU6050 OK" : "MPU6050 FAIL");

  /* ----------- MAX30102 ----------- */
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 FAIL");
    while (1);
  }
  Serial.println("MAX30102 OK");

  // Increase LED brightness + sampling
  particleSensor.setup(0x1F, 4, 2, 100, 0xFF, 0xFF);

  Serial.println("All sensors initialized.\n");

  // Initialize preferences and load stored coefficients (if any)
  // Load network preferences (SSID / PASS / ThingSpeak key) first
  prefs.begin("net", false);
  // Use existing variable values as defaults so hardcoded/test values aren't overwritten by empty NVS
  wifi_ssid = prefs.getString("ssid", wifi_ssid);
  wifi_pass = prefs.getString("pass", wifi_pass);
  thingspeak_key = prefs.getString("ts_key", thingspeak_key);
  prefs.end();

  // If we have credentials, attempt to connect
  if (wifi_ssid.length() > 0) {
    connectWiFi(8000);
  }

  // Print loaded network status (do not print password)
  Serial.println("\n--- Network configuration loaded ---");
  Serial.print("WiFi SSID (loaded): "); Serial.println(wifi_ssid.length() ? wifi_ssid : String("(not set)"));
  Serial.print("ThingSpeak Key set: "); Serial.println(thingspeak_key.length() ? "yes" : "no");
  Serial.println("------------------------------------\n");

  // Now load model prefs
  prefs.begin("model", false);
  
  // Load polynomial coefficients (14 total)
  char key[8];
  for (int i = 0; i < 14; i++) {
    snprintf(key, sizeof(key), "c%d", i);
    COEF_POLY[i] = prefs.getFloat(key, COEF_POLY[i]);
  }
  INTERCEPT_C = prefs.getFloat("ic", INTERCEPT_C);
  MODEL_R2 = prefs.getFloat("r2", MODEL_R2);
  MODEL_RMSE = prefs.getFloat("rmse", MODEL_RMSE);
  // load scaler params if present
  for (int i = 0; i < 4; i++) {
    snprintf(key, sizeof(key), "sm%d", i);
    SCALER_MEAN[i] = prefs.getFloat(key, SCALER_MEAN[i]);
    snprintf(key, sizeof(key), "ss%d", i);
    SCALER_SCALE[i] = prefs.getFloat(key, SCALER_SCALE[i]);
  }
  // load legacy single-feature coefficients if present
  float sentinel = 1e30;
  float ta = prefs.getFloat("a", sentinel);
  if (ta != sentinel) {
    COEF_A = ta;
    COEF_B = prefs.getFloat("b", COEF_B);
    COEF_C = prefs.getFloat("c", COEF_C);
    SCALE_NIR = prefs.getFloat("s", SCALE_NIR);
    use_single_model = true;
  }

  Serial.println("\n========== MODEL LOADED FROM PREFERENCES ==========");
  Serial.print("Model Type: ridge_poly2 (Ridge Polynomial Degree-2)\n");
  Serial.print("R² Score: "); Serial.println(MODEL_R2);
  Serial.print("RMSE: "); Serial.println(MODEL_RMSE);
  Serial.println("Polynomial Coefficients:");
  for (int i = 0; i < 14; i++) {
    Serial.print("  ["); Serial.print(i); Serial.print("] = ");
    Serial.println(COEF_POLY[i]);
  }
  Serial.print("Intercept: "); Serial.println(INTERCEPT_C);
  Serial.println("Scaler mean:");
  for (int i = 0; i < 4; i++) {
    Serial.print("  ["); Serial.print(i); Serial.print("] = ");
    Serial.println(SCALER_MEAN[i]);
  }
  Serial.println("Scaler scale:");
  for (int i = 0; i < 4; i++) {
    Serial.print("  ["); Serial.print(i); Serial.print("] = ");
    Serial.println(SCALER_SCALE[i]);
  }
  Serial.println("=====================================================\n");

  Serial.println("To update model coefficients over Serial, send JSON with:");
  Serial.println("  - \"coef_poly\": [array of 14 coefficients]");
  Serial.println("  - \"intercept_c\": intercept value");
  Serial.println("  - \"r2\": R² performance metric");
  Serial.println("  - \"rmse\": RMSE performance metric");
  Serial.println("\nTo configure WiFi/ThingSpeak via Serial, send JSON with:");
  Serial.println("  - \"ssid\": \"YOUR_SSID\"");
  Serial.println("  - \"pass\": \"YOUR_WIFI_PASSWORD\"");
  Serial.println("  - \"ts_key\": \"THINGSPEAK_WRITE_KEY\"");
  Serial.println("\nOr send 'GET' to print current model coefficients.");
}

/* ---------------- MAIN LOOP ---------------- */
void loop() {

  // Handle incoming serial commands (coefficient updates)
  checkSerialCommands();

  // If WiFi not connected, periodically try to reconnect (if SSID is set)
  if (!wifi_connected && wifi_ssid.length() > 0) {
    if ((millis() - lastWifiTryMillis) > WIFI_RETRY_INTERVAL_MS) {
      Serial.println("WiFi not connected; attempting reconnect...");
      connectWiFi(8000);
      lastWifiTryMillis = millis();
    }
  }


  // Read MAX30102 raw values
  uint32_t rawIR = 0;
  if (particleSensor.check()) {
    rawIR = particleSensor.getIR();  // NIR value
  }

  // Read BME280 temperature
  float temp = NAN;
  if (bme_present) {
    temp = bme.readTemperature();
  } else {
    // attempt quick re-init if previously not present
    if (bme.begin(0x76) || bme.begin(0x77)) {
      bme_present = true;
      Serial.println("BME280 detected during loop re-init");
      temp = bme.readTemperature();
    } else {
      temp = NAN;
    }
  }

  // Read MPU6050 (kept active, NOT printed)
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);  // Not used

  // Print only requested values
  Serial.println("----- SENSOR OUTPUT -----");
  Serial.print("NIR Value: ");
  Serial.println(rawIR);

  Serial.print("Temperature (C): ");
  if (isnan(temp)) {
    Serial.println("-- (BME280 not available)");
  } else {
    Serial.println(temp);
    // Print additional BME280 diagnostics if reading is out of expected range
    if (temp > 80.0 || temp < -40.0) {
      Serial.print("Warning: BME280 temperature out of range: "); Serial.println(temp);
      // print pressure and humidity for debugging
      float pres = bme.readPressure();
      float hum = bme.readHumidity();
      Serial.print("BME280 pressure (Pa): "); Serial.println(pres);
      Serial.print("BME280 humidity (%): "); Serial.println(hum);
      // Try reinitializing once
      if (!bme_present) {
        if (bme.begin(0x76) || bme.begin(0x77)) {
          bme_present = true;
          Serial.println("BME280 reinitialized successfully");
        }
      }
    }
  }

  // Compute calibrated glucose using polynomial ridge regression model
  // Features: AGE, AFTER_FOOD, IS_DIABETIC, NIR_VALUE (normalized)
  float glucose = NAN;
  if (rawIR > 0) {
    // For demonstration, use fixed feature values
    // In a real application, these would come from user input or stored data
    float age = 30.0;          // placeholder
    float afterFood = 1.0;     // 1 = yes, 0 = no
    float isDiabetic = 0.0;    // 1 = yes, 0 = no
    float nirRaw = (float)rawIR;

    // Build raw feature vector
    float rawFeat[4];
    rawFeat[0] = age;
    rawFeat[1] = afterFood;
    rawFeat[2] = isDiabetic;
    rawFeat[3] = nirRaw;

    // Apply scaler: scaled = (raw - mean) / scale
    float s[4];
    for (int i = 0; i < 4; i++) {
      float denom = SCALER_SCALE[i];
      if (denom == 0) denom = 1.0; // avoid div by zero
      s[i] = (rawFeat[i] - SCALER_MEAN[i]) / denom;
    }

    // Build polynomial features array from scaled features
    float polyFeatures[14];
    polyFeatures[0] = s[0];
    polyFeatures[1] = s[1];
    polyFeatures[2] = s[2];
    polyFeatures[3] = s[3];
    polyFeatures[4] = s[0] * s[0];
    polyFeatures[5] = s[0] * s[1];
    polyFeatures[6] = s[0] * s[2];
    polyFeatures[7] = s[0] * s[3];
    polyFeatures[8] = s[1] * s[1];
    polyFeatures[9] = s[1] * s[2];
    polyFeatures[10] = s[1] * s[3];
    polyFeatures[11] = s[2] * s[2];
    polyFeatures[12] = s[2] * s[3];
    polyFeatures[13] = s[3] * s[3];
    
    float glucose = 0.0;
    if (use_single_model) {
      // compute using legacy single-feature polynomial on x = rawIR / SCALE_NIR
      float x = ((float)rawIR) / SCALE_NIR;
      glucose = COEF_A * x * x + COEF_B * x + COEF_C;
    } else {
      // Compute glucose: dot product + intercept
      glucose = INTERCEPT_C;
      for (int i = 0; i < 14; i++) {
        glucose += COEF_POLY[i] * polyFeatures[i];
      }
    }

    Serial.print("Predicted Glucose (mg/dL): ");
    Serial.println(glucose);
    // If enough time has elapsed since last ThingSpeak update, send data
    if ((millis() - lastThingSpeakMillis) > THINGSPEAK_INTERVAL_MS) {
      sendToThingSpeak(glucose, temp, rawIR);
      lastThingSpeakMillis = millis();
    }
  } else {
    Serial.println("Predicted Glucose (mg/dL): --");
  }

  Serial.println("--------------------------\n");

  delay(1000);
}
