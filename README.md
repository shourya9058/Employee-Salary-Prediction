# Salary Prediction Web Application

A machine learning web application that predicts income levels based on demographic and employment data using a Random Forest Classifier.

## Features

- **Predict Income**: Input demographic and employment details to predict income levels
- **Model Training**: Upload custom datasets to train and improve the model
- **Responsive Design**: Works on both desktop and mobile devices
- **Real-time Feedback**: Get immediate predictions with confidence scores
- **Model Management**: View model status, accuracy, and retrain with new data

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/salary-prediction-app.git
   cd salary-prediction-app
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running Locally

1. **Start the Flask development server**
   ```bash
   python app.py
   ```

2. **Open your browser**
   Visit `http://localhost:5000` to access the application

## Deployment

This application is configured for deployment on [Render](https://render.com/).

### Deploy to Render

1. **Create a new Web Service** on Render
2. **Connect your GitHub/GitLab repository** or deploy using the Render Dashboard
3. **Configure the following settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --worker-tmp-dir /dev/shm --workers 2 --threads 4 --bind 0.0.0.0:$PORT app:app`
   - **Environment Variables**:
     - `PYTHON_VERSION=3.9.16`
     - `FLASK_ENV=production`

4. **Deploy!**

## Project Structure

```
salary-prediction-app/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── runtime.txt           # Python version specification
├── render.yaml           # Render deployment configuration
├── Procfile             # Process file for Gunicorn
├── static/              # Static files (CSS, JS, images)
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # Styling
├── templates/           # HTML templates
│   └── index.html       # Main application page
└── uploads/             # Directory for uploaded datasets (created at runtime)
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── static/              # Static files (CSS, JS, images)
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # Custom styles
├── templates/           # HTML templates
│   └── index.html       # Main application page
└── adult 3.csv          # Sample dataset
```

## API Documentation

### Endpoints

#### 1. Predict Income

* **URL**: `/predict`
* **Method**: `POST`
* **Request Body**: JSON object with demographic and employment details
* **Response**: JSON object with predicted income level and confidence score

#### 2. Train Model

* **URL**: `/train`
* **Method**: `POST`
* **Request Body**: JSON object with dataset details
* **Response**: JSON object with model status and accuracy

#### 3. Get Model Status

* **URL**: `/model-status`
* **Method**: `GET`
* **Response**: JSON object with model status and accuracy

## Model

The application uses a Random Forest Classifier for salary prediction. The model is trained on the following features:

- Age
- Work Class
- Education
- Occupation
- Marital Status
- Hours per Week

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
