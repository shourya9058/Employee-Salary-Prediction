from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import traceback
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Set the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set maximum upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables to store model and encoders
model = None
label_encoders = {}

# Initialize or load model
def init_model():
    global model, label_encoders, model_metadata, model_trained_on, model_accuracy
    model = None
    label_encoders = {}
    model_metadata = {}
    model_trained_on = None
    model_accuracy = None
    
    try:
        model_path = 'model.joblib'
        encoders_path = 'label_encoders.joblib'
        metadata_path = 'model_metadata.joblib'
        
        # Check if model files exist
        if all(os.path.exists(p) for p in [model_path, encoders_path, metadata_path]):
            try:
                model = joblib.load(model_path)
                label_encoders = joblib.load(encoders_path)
                model_metadata = joblib.load(metadata_path)
                
                # Set global variables from metadata
                model_trained_on = model_metadata.get('trained_on')
                model_accuracy = model_metadata.get('accuracy')
                
                print(f"Model loaded successfully. Last trained on: {model_trained_on}")
                return "Ready"
                
            except Exception as e:
                print(f"Error loading model files: {str(e)}")
                print(traceback.format_exc())
                return "Error"
                
        print("No trained model found. Please train a model first.")
        return "Not Trained"
        
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print(traceback.format_exc())
        return "Error"

def preprocess_data(df):
    """Preprocess the input dataframe."""
    # Handle missing values
    df = df.replace(' ?', np.nan)
    df = df.dropna()
    
    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()
    
    # Clean string columns (remove extra spaces)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

def train_model(filepath):
    """Train the model with the given dataset file."""
    global model, label_encoders, model_trained_on, model_accuracy
    
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Preprocess the data
        df = preprocess_data(df)
        
        # Check if required columns exist
        required_columns = [
            'age', 'workclass', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain',
            'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Split features and target
        X = df.drop('income', axis=1)
        y = df['income']
        
        # Encode categorical variables
        label_encoders = {}
        for column in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model with balanced class weights
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy and other metrics
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get feature importances
        feature_importances = dict(zip(X.columns, model.feature_importances_))
        
        # Get current timestamp
        trained_on = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare metadata
        metadata = {
            'trained_on': trained_on,
            'accuracy': accuracy,
            'samples': {
                'total': len(df),
                'train': len(X_train),
                'test': len(X_test)
            },
            'class_distribution': dict(y.value_counts()),
            'classification_report': report,
            'feature_importances': feature_importances
        }
        
        # Update global variables
        model_trained_on = trained_on
        model_accuracy = accuracy
        
        return model, label_encoders, metadata
        
    except Exception as e:
        import traceback
        error_msg = f"Error in train_model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise ValueError(f"Error training model: {str(e)}")

# Routes
@app.route('/')
def home():
    return render_template('index.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Model status endpoint
@app.route('/status', methods=['GET'])
def get_model_status():
    try:
        status = init_model()
        
        # Prepare basic status
        response = {
            'status': status,
            'trained_on': model_trained_on,
            'accuracy': model_accuracy,
        }
        
        # Add feature information if model is loaded
        if label_encoders:
            response['features'] = list(label_encoders.keys())
            
        # Add metadata if available
        if model_metadata:
            response['metadata'] = {
                'samples': model_metadata.get('samples', {}),
                'class_distribution': model_metadata.get('class_distribution', {}),
                'feature_importances': model_metadata.get('feature_importances', {})
            }
            
            # Add classification report if available
            if 'classification_report' in model_metadata:
                report = model_metadata['classification_report']
                if 'weighted avg' in report:
                    response['metrics'] = {
                        'precision': report['weighted avg']['precision'],
                        'recall': report['weighted avg']['recall'],
                        'f1_score': report['weighted avg']['f1-score'],
                        'support': report['weighted avg']['support']
                    }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'Error',
            'error': f'Error getting model status: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
            
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file and file.filename.lower().endswith('.csv'):
            try:
                # Create uploads directory if it doesn't exist
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                
                # Save the uploaded file with a secure filename
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read the CSV to validate it
                try:
                    df = pd.read_csv(filepath)
                    if len(df) == 0:
                        os.remove(filepath)  # Clean up empty file
                        return jsonify({'error': 'The uploaded CSV file is empty'}), 400
                except Exception as e:
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Clean up invalid file
                    return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
                
                # Train the model with the new dataset
                global model, label_encoders, model_metadata, model_trained_on, model_accuracy
                
                try:
                    model, label_encoders, metadata = train_model(filepath)
                    
                    # Update global variables
                    model_metadata = metadata
                    model_trained_on = metadata['trained_on']
                    model_accuracy = metadata['accuracy']
                    
                    # Save the trained model and encoders
                    joblib.dump(model, 'model.joblib')
                    joblib.dump(label_encoders, 'label_encoders.joblib')
                    joblib.dump(metadata, 'model_metadata.joblib')
                    
                    # Prepare response
                    response = {
                        'message': 'Model trained successfully',
                        'metadata': {
                            'trained_on': model_trained_on,
                            'accuracy': model_accuracy,
                            'samples': metadata['samples'],
                            'class_distribution': metadata['class_distribution']
                        }
                    }
                    
                    return jsonify(response)
                    
                except ValueError as ve:
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Clean up on error
                    return jsonify({
                        'error': f'Training error: {str(ve)}',
                        'details': str(ve)
                    }), 400
                    
                except Exception as e:
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Clean up on error
                    return jsonify({
                        'error': f'Error during model training: {str(e)}',
                        'details': traceback.format_exc()
                    }), 500
                
            except Exception as e:
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)  # Clean up on error
                return jsonify({
                    'error': f'Error processing file: {str(e)}',
                    'details': traceback.format_exc()
                }), 500
                
        else:
            return jsonify({'error': 'Only CSV files are allowed'}), 400
            
    except Exception as e:
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'details': traceback.format_exc()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request is JSON
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'error': 'Request must be JSON',
                'details': 'Content-Type must be application/json'
            }), 400
            
        data = request.get_json()
        
        # Check if model is loaded
        if model is None or not label_encoders:
            init_status = init_model()
            if init_status != 'Ready':
                return jsonify({
                    'status': 'error',
                    'error': 'Model not ready',
                    'details': f'Model status: {init_status}'
                }), 503  # Service Unavailable
        
        # Validate input data
        if not isinstance(data, dict):
            return jsonify({
                'status': 'error',
                'error': 'Invalid input format',
                'details': 'Input must be a JSON object with feature values'
            }), 400
        
        # Check for missing features
        required_features = set(label_encoders.keys()) - {'income'}
        missing_features = required_features - set(data.keys())
        if missing_features:
            return jsonify({
                'status': 'error',
                'error': 'Missing required features',
                'details': f'Missing features: {sorted(missing_features)}',
                'required_features': sorted(required_features)
            }), 400
        
        try:
            # Prepare input data
            input_data = {}
            for feature in required_features:
                value = data[feature]
                
                # Convert numeric strings to appropriate types
                if isinstance(value, str):
                    if value.replace('.', '', 1).isdigit():
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                
                input_data[feature] = [value]
            
            # Create DataFrame
            df = pd.DataFrame(input_data)
            
            # Encode categorical variables
            for col in df.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    try:
                        # Handle unseen labels by encoding them as -1
                        df[col] = df[col].apply(
                            lambda x: label_encoders[col].transform([str(x)])[0] 
                            if str(x) in label_encoders[col].classes_ 
                            else -1
                        )
                    except Exception as e:
                        return jsonify({
                            'status': 'error',
                            'error': f'Error encoding feature: {col}',
                            'details': str(e),
                            'allowed_values': list(label_encoders[col].classes_)
                        }), 400
            
            # Make prediction
            prediction = model.predict(df)
            probability = model.predict_proba(df)
            
            # Get class labels
            if hasattr(label_encoders.get('income'), 'classes_'):
                class_labels = label_encoders['income'].classes_
                predicted_class = class_labels[prediction[0]]
                
                # Map probabilities to class labels
                probabilities = {
                    str(cls): float(prob) 
                    for cls, prob in zip(class_labels, probability[0])
                }
            else:
                predicted_class = str(prediction[0])
                probabilities = {}
            
            # Calculate confidence
            confidence = float(np.max(probability[0]))
            
            # Prepare response
            response = {
                'status': 'success',
                'prediction': predicted_class,
                'confidence': round(confidence, 4),
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat(),
                'model': {
                    'trained_on': model_trained_on,
                    'accuracy': round(model_accuracy, 4) if model_accuracy is not None else None
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': 'Prediction failed',
                'details': str(e),
                'traceback': traceback.format_exc()
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'Unexpected error',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Initialize the model when the app starts
init_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
