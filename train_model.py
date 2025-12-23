import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import argparse
import time

CSV_PATH = 'NIR_Blood_Glucose.csv'
SCALE = 1e5

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None

def load_data(path):
    df = pd.read_csv(path)
    # Keep only first 120 rows (or available rows up to 120)
    df = df.head(120)
    # Use only NIR VALUE -> GLUCOSE(mg/dL)
    df = df.dropna(subset=['NIR VALUE', 'GLUCOSE(mg/dL)'])
    X = df['NIR VALUE'].values.reshape(-1,1).astype(float)
    y = df['GLUCOSE(mg/dL)'].values.astype(float)
    print(f"Dataset loaded: {len(X)} samples, 1 feature (NIR VALUE)")
    return X, y

def fit_models(X, y):
    # For single-feature (NIR) models we will normalize by a fixed SCALE
    Xs = X / SCALE

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

    results = {}

    # Model 1: Linear Regression (on normalized NIR)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results['linear'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'coef': lr.coef_.flatten().tolist(),
        'intercept': float(lr.intercept_)
    }

    # Model 2: Polynomial degree 2
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp_train = poly.fit_transform(X_train)
    Xp_test = poly.transform(X_test)
    lr2 = LinearRegression()
    lr2.fit(Xp_train, y_train)
    y2_pred = lr2.predict(Xp_test)
    # coef order corresponds to [x, x^2]
    results['poly2'] = {
        'rmse': np.sqrt(mean_squared_error(y_test, y2_pred)),
        'mae': mean_absolute_error(y_test, y2_pred),
        'r2': r2_score(y_test, y2_pred),
        'coef': lr2.coef_.flatten().tolist(),
        'intercept': float(lr2.intercept_)
    }

    return results, lr, lr2

def choose_and_save(results, scaler=None):
    # Choose model with highest R² (best fit) and lowest RMSE
    print('\n' + '='*60)
    print('MODEL COMPARISON RESULTS')
    print('='*60)
    print(json.dumps(results, indent=2))
    
    # Rank models by R² score (primary metric for accuracy)
    ranked = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print('\n' + '='*60)
    print('RANKING BY R² (Coefficient of Determination):')
    print('='*60)
    for i, (name, metrics) in enumerate(ranked, 1):
        print(f"{i}. {name:20s} - R²: {metrics['r2']:.6f}, RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    chosen = ranked[0][0]
    print(f'\n[SELECTED] CHOSEN MODEL: {chosen}')
    print(f"  R² Score: {results[chosen]['r2']:.6f}")
    print(f"  RMSE: {results[chosen]['rmse']:.4f} mg/dL")
    print(f"  MAE: {results[chosen]['mae']:.4f} mg/dL")
    # No CV R² for single-feature models in this simplified pipeline
    
    # Save a legacy single-feature format for compatibility with the ESP32 sketch
    out = {
        'model': chosen,
        'feature': 'NIR VALUE',
        'performance': {
            'r2': results[chosen]['r2'],
            'rmse': results[chosen]['rmse'],
            'mae': results[chosen]['mae']
        }
    }
    
    # Add poly features info if applicable
    if chosen == 'linear':
        # coef list contains single element for NIR
        b = results[chosen]['coef'][0]
        c = results[chosen]['intercept']
        out.update({'coef_a': None, 'coef_b': b, 'intercept_c': c, 'scale': int(SCALE)})
    else:
        # poly2: coef order is [x, x^2]
        b = results[chosen]['coef'][0]
        a = results[chosen]['coef'][1]
        c = results[chosen]['intercept']
        out.update({'coef_a': a, 'coef_b': b, 'intercept_c': c, 'scale': int(SCALE)})
    
    with open('model_coeffs.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    
    print(f'\n[SAVED] model_coeffs.json')
    return chosen

def main():
    parser = argparse.ArgumentParser(description='Train calibration model on all features and optionally push to ESP32')
    parser.add_argument('--port', help='Serial port to upload model (e.g., COM3). If omitted no upload')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate for serial upload')
    parser.add_argument('--wait-response', action='store_true', help='Wait and print device response after sending')
    parser.add_argument('--dry-run-upload', action='store_true', help='Do not actually open serial port; print payload')
    parser.add_argument('--auto-upload', action='store_true', help='Automatically detect ESP32 port and upload')
    args = parser.parse_args()

    print("Loading and training models...")
    X, y = load_data(CSV_PATH)
    results, lr, lr2 = fit_models(X, y)
    chosen_model = choose_and_save(results)

    # Load the saved model for validation on entire dataset
    with open('model_coeffs.json', 'r', encoding='utf-8') as f:
        m = json.load(f)
    
    print("\n" + "="*60)
    print("FULL DATASET VALIDATION:")
    print("="*60)
    
    # Validate on full dataset using legacy single-feature format (NIR normalized by SCALE)
    Xs = X / SCALE
    if m.get('coef_a') is None:
        # linear model
        b = m.get('coef_b', 0)
        c = m.get('intercept_c', 0)
        yhat = b * Xs.flatten() + c
    else:
        a = m.get('coef_a', 0)
        b = m.get('coef_b', 0)
        c = m.get('intercept_c', 0)
        yhat = a * (Xs.flatten()**2) + b * Xs.flatten() + c
    
    rmse = np.sqrt(mean_squared_error(y, yhat))
    mae = mean_absolute_error(y, yhat)
    r2 = r2_score(y, yhat)
    
    print(f"Full-dataset R-Squared: {r2:.6f}")
    print(f"Full-dataset RMSE: {rmse:.4f} mg/dL")
    print(f"Full-dataset MAE: {mae:.4f} mg/dL")
    
    # Auto-detect and upload if requested
    if args.auto_upload:
        if serial is None:
            print('\nERROR: pyserial not available; install with pip install pyserial')
            return
        
        # Try to find ESP32 port
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            print('\nERROR: No serial ports found')
            return
        
        # Use first available port (typically ESP32)
        port = ports[0]
        print(f"\nAuto-detected port: {port}")
        args.port = port
    
    # If requested, upload to serial port
    if args.port:
        # Build payload using legacy single-feature keys (coef_a, coef_b, intercept_c, scale)
        payload_dict = {
            'model': m.get('model'),
            'coef_a': m.get('coef_a'),
            'coef_b': m.get('coef_b'),
            'intercept_c': m.get('intercept_c'),
            'scale': int(m.get('scale', SCALE)),
            'r2': m['performance']['r2'],
            'rmse': m['performance']['rmse'],
            'mae': m['performance']['mae']
        }
        payload = json.dumps(payload_dict, separators=(',', ':')) + '\n'

        print('\nPayload to send:')
        print(payload)

        if args.dry_run_upload:
            print('Dry-run upload; not opening serial port.')
            return

        if serial is None:
            print('pyserial not available; install with pip install pyserial')
            return

        try:
            ser = serial.Serial(args.port, args.baud, timeout=2)
        except OSError as e:
            print('Failed to open serial port:', e)
            return

        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        ser.write(payload.encode())
        ser.flush()
        print(f'Payload sent to {args.port}')

        if args.wait_response:
            time.sleep(0.2)
            out = b''
            while True:
                chunk = ser.read(1024)
                if not chunk:
                    break
                out += chunk
            try:
                print('Device reply:\n', out.decode(errors='ignore'))
            except Exception as e:
                print('Device reply (raw):', out, e)

        ser.close()

if __name__ == '__main__':
    main()
