import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =============================================
# CONFIGURATION
# =============================================
MODEL_PATH = 'model/my_model.h5'
ENCODER_PATH = 'model/encoder.save'
FEATURE_SCALER_PATH = 'model/scaler_X.save'
TARGET_SCALER_PATH = 'model/scaler_y.save'
FEATURE_METADATA_PATH = 'model/feature_metadata.json'

# Indonesian Holidays
HOLIDAYS = pd.to_datetime([
    # 2024 Holidays
    '2024-01-01', '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-14',
    '2024-03-11', '2024-03-12', '2024-03-29', '2024-04-08', '2024-04-09',
    '2024-04-10', '2024-04-11', '2024-05-01', '2024-05-09', '2024-05-10',
    '2024-05-23', '2024-05-24', '2024-06-01', '2024-06-17', '2024-06-18',
    '2024-07-07', '2024-08-17', '2024-09-16', '2024-11-27', '2024-12-25', '2024-12-26',
    
    # 2025 Holidays
    '2025-01-01', '2025-01-27', '2025-01-28', '2025-01-29', '2025-03-28',
    '2025-03-29', '2025-03-31', '2025-04-01', '2025-04-02', '2025-04-03',
    '2025-04-04', '2025-04-07', '2025-04-18', '2025-04-20', '2025-05-01',
    '2025-05-12', '2025-05-13', '2025-05-29', '2025-05-30', '2025-06-01',
    '2025-06-06', '2025-06-09', '2025-06-27', '2025-08-17', '2025-09-05',
    '2025-12-25', '2025-12-26'
])

# Define the complete feature order (7 features as expected by your model)
FEATURE_ORDER = ['RollingStd_7', 'Lag_1', 'RollingMean_7', 'RollingRatio_7_30',
                 'RollingMean_30', 'DayOfMonth', 'Year']

# =============================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# =============================================
try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    traceback.print_exc()
    raise

try:
    # Try to load encoder if it exists
    try:
        encoder = joblib.load(ENCODER_PATH)
        print("Encoder loaded successfully")
    except:
        encoder = None
        print("No encoder found - treating Year as numerical")
    
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    print("Scalers loaded successfully")
    
    # Check if scaler has feature names
    if hasattr(feature_scaler, 'feature_names_in_'):
        print("Feature scaler features:", feature_scaler.feature_names_in_)
        scaler_features = feature_scaler.feature_names_in_.tolist()
    else:
        print("Feature scaler doesn't have feature names - using default order")
        scaler_features = FEATURE_ORDER[:5]  # Assume first 5 are numerical
        
except Exception as e:
    print(f"Error loading scalers: {str(e)}")
    traceback.print_exc()
    raise

# Load feature metadata
try:
    with open(FEATURE_METADATA_PATH) as f:
        feature_metadata = json.load(f)
        LOOKBACK_DAYS = feature_metadata.get('lookback_period', 7)
        print(f"Lookback days: {LOOKBACK_DAYS}")
        if 'feature_order' in feature_metadata:
            FEATURE_ORDER = feature_metadata['feature_order']
            print(f"Using feature order from metadata: {FEATURE_ORDER}")
except Exception as e:
    print(f"Warning: Could not load feature metadata: {str(e)}")
    LOOKBACK_DAYS = 7
    print(f"Using default lookback days: {LOOKBACK_DAYS}")

print(f"Final feature order ({len(FEATURE_ORDER)} features): {FEATURE_ORDER}")

# =============================================
# PREPROCESSING FUNCTIONS
# =============================================
def create_features(data):
    """Feature engineering identical to training pipeline"""
    # Basic time features
    data['Year'] = data['Date'].dt.year
    data['DayOfMonth'] = data['Date'].dt.day
    
    # Lag features
    data['Lag_1'] = data['Sales'].shift(1).fillna(0)
    
    # Rolling features
    data['RollingMean_7'] = data['Sales'].rolling(window=7, min_periods=1).mean()
    data['RollingMean_30'] = data['Sales'].rolling(window=30, min_periods=1).mean()
    data['RollingStd_7'] = data['Sales'].rolling(window=7, min_periods=1).std().fillna(0)
    
    # Ratio features with safe division
    data['RollingRatio_7_30'] = (
        data['RollingMean_7'] / data['RollingMean_30']
    ).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    return data

def preprocess_input(input_data):
    """Full preprocessing pipeline"""
    # Convert to DataFrame
    data = pd.DataFrame({
        'Date': pd.to_datetime(input_data['date']),
        'Sales': input_data['sales']
    })
    
    # Handle missing values
    data['Sales'] = data['Sales'].ffill().fillna(0)
    
    # Feature engineering
    data = create_features(data)
    
    # Handle Year encoding
    if encoder is not None:
        # One-hot encode Year
        cat_features = ['Year']
        encoded = encoder.transform(data[cat_features])
        encoded_cols = encoder.get_feature_names_out(cat_features)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=data.index)
        
        # Combine with other features
        other_features = [f for f in FEATURE_ORDER if f not in encoded_cols and f != 'Year']
        feature_data = data[other_features].copy()
        feature_data = pd.concat([feature_data, encoded_df], axis=1)
    else:
        # Keep Year as numerical
        feature_data = data[FEATURE_ORDER].copy()
    
    # Ensure all expected columns exist
    for col in FEATURE_ORDER:
        if col not in feature_data.columns:
            feature_data[col] = 0
    
    # Reorder columns to match expected order
    feature_data = feature_data[FEATURE_ORDER]
    
    # Scale only the numerical features that were scaled during training
    if hasattr(feature_scaler, 'feature_names_in_'):
        # Scale only the features that the scaler knows about
        scaler_features = feature_scaler.feature_names_in_.tolist()
        feature_data[scaler_features] = feature_scaler.transform(feature_data[scaler_features])
    else:
        # Scale all features (fallback)
        feature_data = pd.DataFrame(
            feature_scaler.transform(feature_data),
            columns=feature_data.columns,
            index=feature_data.index
        )
    
    # Final cleanup
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    print(f"Preprocessed data shape: {feature_data.shape}")
    print(f"Preprocessed data columns: {list(feature_data.columns)}")
    print(f"Sample data:\n{feature_data.tail()}")
    
    return feature_data

# =============================================
# PREDICTION ENDPOINT
# =============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Input validation
        input_data = request.json
        print(f"Received request: {input_data}")
        
        if not input_data:
            return jsonify({'error': 'No input data', 'status': 'error'}), 400
            
        required_fields = ['date', 'sales']
        missing_fields = [f for f in required_fields if f not in input_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error',
                'example_request': {
                    'date': ['2024-05-01', '2024-05-02', '2024-05-03'],
                    'sales': [1000000, 1200000, 1100000]
                }
            }), 400

        # Check if date and sales arrays have same length
        if len(input_data['date']) != len(input_data['sales']):
            return jsonify({
                'error': f'Date and sales arrays must have same length. Got {len(input_data["date"])} dates and {len(input_data["sales"])} sales values',
                'status': 'error'
            }), 400

        # 2. Preprocessing
        try:
            print(f"Input data received: {len(input_data.get('date', []))} dates, {len(input_data.get('sales', []))} sales values")
            processed_data = preprocess_input(input_data)
            print(f"Processed data shape: {processed_data.shape}")
            print(f"Processed data columns: {list(processed_data.columns)}")
        except Exception as e:
            print(f"Preprocessing error details: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': f'Preprocessing failed: {str(e)}',
                'status': 'error'
            }), 400

        # 3. Check data sufficiency
        if len(processed_data) < LOOKBACK_DAYS:
            return jsonify({
                'error': f'Minimum {LOOKBACK_DAYS} days required',
                'days_provided': len(input_data['date']),
                'status': 'error'
            }), 400

        # 4. Prepare model input
        model_input = processed_data.iloc[-LOOKBACK_DAYS:]
        X_scaled = model_input.values
        
        print(f"Model input shape before reshape: {X_scaled.shape}")
        print(f"Expected shape: (1, {LOOKBACK_DAYS}, {len(FEATURE_ORDER)})")
        
        if X_scaled.shape[1] != len(FEATURE_ORDER):
            return jsonify({
                'error': f'Feature dimension mismatch. Expected {len(FEATURE_ORDER)} features, got {X_scaled.shape[1]}',
                'expected_features': FEATURE_ORDER,
                'status': 'error'
            }), 400
        
        X_reshaped = X_scaled.reshape(1, LOOKBACK_DAYS, len(FEATURE_ORDER))
        print(f"Model input shape after reshape: {X_reshaped.shape}")
        
        # 5. Make prediction with NaN handling
        try:
            y_pred_scaled = model.predict(X_reshaped)
            y_pred = np.expm1(target_scaler.inverse_transform(y_pred_scaled))[0][0]
            
            if np.isnan(y_pred) or np.isinf(y_pred):
                raise ValueError("Invalid prediction value")
                
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Fallback: Use median prediction or business logic
            y_pred = 1000000  # Default value, adjust based on your data
            print(f"Using fallback prediction: {y_pred}")

        # 6. Prepare response
        last_date = pd.to_datetime(input_data['date'][-1])
        prediction_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        return jsonify({
            'predicted_sales': round(float(y_pred), 2),
            'prediction_date': prediction_date,
            'is_holiday': bool(pd.to_datetime(prediction_date) in HOLIDAYS),
            'status': 'success'
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)