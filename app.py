from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from datetime import datetime

app = Flask(__name__)

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
        # Check if 'income' column exists
        if 'income' not in df.columns:
            return {"error": "Dataset must contain an 'income' column"}
            
        # Handle missing values
        df.replace(' ?', np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Label Encoding
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Prepare data
        X = df.drop("income", axis=1)
        y = df["income"]
        
        # Split data for accuracy calculation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        # Save model, encoders, and metadata
        joblib.dump(model, 'model.joblib')
        joblib.dump(label_encoders, 'encoders.joblib')
        
        # Save metadata
        metadata = {
            'trained_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'accuracy': accuracy,
            'features': list(X.columns)
        }
        joblib.dump(metadata, 'model_metadata.joblib')
        
        # Update global variables
        model_trained_on = metadata['trained_on']
        model_accuracy = accuracy
        
        return {
            "message": "Model trained and saved successfully",
            "accuracy": accuracy,
            "trained_on": model_trained_on
        }
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return {"error": f"Error training model: {str(e)}"}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Model status endpoint
@app.route('/model-status')
def get_model_status():
    status = init_model()
    return jsonify({
        'status': status,
        'trained_on': model_trained_on,
        'accuracy': model_accuracy,
        'features': list(label_encoders.keys()) if label_encoders else []
    })

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload a CSV file.'}), 400
        
        # Train the model
        result = train_model(df)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        return jsonify({
            'message': result['message'],
            'accuracy': result['accuracy'],
            'trained_on': result['trained_on']
        })
    except Exception as e:
        print(f"Error in train endpoint: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
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
