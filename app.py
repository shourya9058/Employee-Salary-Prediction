from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)

app = Flask(__name__)
app.logger.addHandler(handler)

# Global variables to store model and encoders
model = None
label_encoders = {}

# Initialize or load model
def init_model():
    global model, label_encoders, model_trained_on, model_accuracy
    model_trained_on = None
    model_accuracy = None
    
    try:
        if os.path.exists('model.joblib') and os.path.exists('encoders.joblib'):
            model = joblib.load('model.joblib')
            label_encoders = joblib.load('encoders.joblib')
            
            # Load metadata if exists
            if os.path.exists('model_metadata.joblib'):
                metadata = joblib.load('model_metadata.joblib')
                model_trained_on = metadata.get('trained_on')
                model_accuracy = metadata.get('accuracy')
            
            return "Ready"
        return "Not Trained"
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return "Error"

# Train model function
def train_model(df):
    global model, label_encoders, model_trained_on, model_accuracy
    
    try:
        app.logger.info("Starting model training...")
        
        # Check if 'income' column exists
        if 'income' not in df.columns:
            error_msg = "Dataset must contain an 'income' column"
            app.logger.error(error_msg)
            return {"error": error_msg}
        
        # Make a copy to avoid modifying the original dataframe
        df_clean = df.copy()
        
        # Handle missing values
        df_clean.replace('?', np.nan, inplace=True)
        df_clean.replace(' ?', np.nan, inplace=True)
        df_clean.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        
        # Drop rows with missing values
        initial_rows = len(df_clean)
        df_clean.dropna(inplace=True)
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            app.logger.info(f"Dropped {dropped_rows} rows with missing values")
        
        if len(df_clean) == 0:
            error_msg = "No valid data remaining after cleaning"
            app.logger.error(error_msg)
            return {"error": error_msg}
        
        # Convert all object columns to string and trim whitespace
        for col in df_clean.select_dtypes(include='object').columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Label Encoding
        label_encoders = {}
        try:
            for col in df_clean.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                label_encoders[col] = le
        except Exception as e:
            error_msg = f"Error in label encoding: {str(e)}"
            app.logger.error(error_msg)
            return {"error": error_msg}
        
        try:
            # Prepare data
            X = df_clean.drop("income", axis=1)
            y = df_clean["income"]
            
            # Check if we have enough data
            if len(X) < 10:
                error_msg = "Not enough data for training (minimum 10 samples required)"
                app.logger.error(error_msg)
                return {"error": error_msg}
            
            # Split data for accuracy calculation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model with error handling
            try:
                # Create a simpler model configuration to avoid version conflicts
                model = RandomForestClassifier(
                    n_estimators=50,  # Reduced number of estimators for faster training
                    max_depth=10,     # Limit tree depth
                    min_samples_split=5,  # Minimum samples required to split a node
                    min_samples_leaf=2,   # Minimum samples required at a leaf node
                    random_state=42,
                    n_jobs=1,        # Set to 1 to avoid parallel processing issues
                    verbose=0         # Disable verbose output to prevent logging issues
                )
                
                # Fit the model with progress updates
                model.fit(X_train, y_train)
                
                # Calculate accuracy
                accuracy = model.score(X_test, y_test)
                app.logger.info(f"Model trained with accuracy: {accuracy:.4f}")
                
                # Save model, encoders, and metadata
                joblib.dump(model, 'model.joblib')
                joblib.dump(label_encoders, 'encoders.joblib')
                
                # Save metadata
                metadata = {
                    'trained_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'accuracy': float(accuracy),  # Convert numpy.float64 to Python float
                    'features': list(X.columns)
                }
                joblib.dump(metadata, 'model_metadata.joblib')
                
                # Update global variables
                model_trained_on = metadata['trained_on']
                model_accuracy = accuracy
                
                return {
                    "message": "Model trained and saved successfully",
                    "accuracy": float(accuracy),
                    "trained_on": model_trained_on
                }
                
            except Exception as e:
                error_msg = f"Error in model training: {str(e)}"
                app.logger.error(error_msg, exc_info=True)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error in data preparation: {str(e)}"
            app.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Unexpected error in train_model: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

# Serve index.html for all other routes to support client-side routing
@app.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')

# Test route to verify server is working
@app.route('/api/test')
def test():
    return jsonify({
        'status': 'ok',
        'python_version': '3.10.8',
        'flask_version': '2.0.1',
        'model_loaded': model is not None,
        'trained_on': model_trained_on
    })

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Model status endpoint
@app.route('/api/model-status')
def get_model_status():
    status = init_model()
    return jsonify({
        'status': status,
        'trained_on': model_trained_on,
        'accuracy': model_accuracy,
        'features': list(label_encoders.keys()) if label_encoders else []
    })

@app.route('/api/train', methods=['POST'])
def train():
    # Check if the post request has the file part
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request. Please select a file to upload.'
        }), 400
    
    file = request.files['file']
    
    # If the user does not select a file, the browser submits an empty file
    if file.filename == '':
        app.logger.error('No file selected')
        return jsonify({
            'status': 'error',
            'message': 'No file selected. Please select a CSV file to upload.'
        }), 400
    
    # Check if the file has an allowed extension
    allowed_extensions = {'csv'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        app.logger.error(f'Invalid file type: {file.filename}')
        return jsonify({
            'status': 'error',
            'message': 'Unsupported file format. Please upload a CSV file with .csv extension.'
        }), 400
    
    # Check file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file.seek(0, 2)  # Go to end of file
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size > max_size:
        app.logger.error(f'File too large: {file_size} bytes')
        return jsonify({
            'status': 'error',
            'message': f'File is too large. Maximum size is {max_size//(1024*1024)}MB.'
        }), 400
    
    try:
        app.logger.info(f'Processing file: {file.filename}')
        
        # Read the file into a pandas DataFrame
        try:
            # Try different encodings if the default fails
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                file.seek(0)
                df = pd.read_csv(file, encoding='latin1')
                
            app.logger.info(f'Successfully read CSV file with shape: {df.shape}')
            
            # Basic data validation
            if df.empty:
                raise ValueError('The uploaded file is empty')
                
            # Check for required columns
            if 'income' not in df.columns:
                raise ValueError("CSV file must contain an 'income' column")
                
            # Check for minimum data requirements
            if len(df) < 10:
                raise ValueError("Not enough data for training (minimum 10 samples required)")
                
        except Exception as e:
            error_msg = f'Error reading or validating CSV file: {str(e)}'
            app.logger.error(error_msg)
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400
        
        # Train the model
        app.logger.info('Starting model training...')
        result = train_model(df)
        
        if 'error' in result:
            app.logger.error(f'Error in model training: {result["error"]}')
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 400
        
        app.logger.info(f'Model training completed successfully. Accuracy: {result["accuracy"]:.2f}')
            
        return jsonify({
            'status': 'success',
            'message': result['message'],
            'accuracy': result['accuracy'],
            'trained_on': result['trained_on']
        })
        
    except Exception as e:
        error_msg = f'Unexpected error in train endpoint: {str(e)}'
        app.logger.error(error_msg, exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred while processing your request. Please try again.'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    # Initialize model if not already done
    init_model()
    
    if model is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([data])
        
        # Apply label encoding to categorical columns
        for col in input_data.select_dtypes(include='object').columns:
            if col in label_encoders:
                try:
                    # Handle unseen labels
                    input_data[col] = input_data[col].apply(
                        lambda x: x if x in label_encoders[col].classes_ else None
                    )
                    input_data = input_data.dropna()  # Drop rows with unknown labels
                    if len(input_data) == 0:
                        return jsonify({'error': f'Unknown value for {col}. Please check your input.'}), 400
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except Exception as e:
                    return jsonify({'error': f'Error processing {col}: {str(e)}'}), 400
        
        # Ensure all required features are present
        missing_features = set(model.feature_names_in_) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}'
            }), 400
        
        # Reorder columns to match training data
        input_data = input_data[model.feature_names_in_]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get the predicted class and its probability
        predicted_class = int(prediction)
        confidence = float(probabilities.max())
        
        # Convert prediction back to original label if possible
        if 'income' in label_encoders:
            predicted_label = label_encoders['income'].inverse_transform([predicted_class])[0]
        else:
            predicted_label = '>50K' if predicted_class == 1 else '<=50K'
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'class': predicted_class
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    init_model()
    app.run(debug=True)
