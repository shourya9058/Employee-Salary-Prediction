# Salary Prediction System

A web-based application that predicts employee salary ranges based on various demographic and employment factors using machine learning.

## Features

- **Interactive Form**: Input employee details to get instant salary predictions
- **Model Training**: Upload new datasets to train and improve the prediction model
- **Responsive Design**: Works on both desktop and mobile devices
- **Visual Feedback**: Clear visualizations of prediction results and model performance
- **Drag & Drop**: Easy file upload interface with drag and drop support

## Local Development

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   cd salary-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Start the Flask development server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the form to input employee details and get salary predictions, or upload a new dataset to train the model.

## Deployment to Render

1. **Create a new Web Service** on Render and connect to your GitHub/GitLab repository.

2. Configure the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Python Version**: 3.8 or higher

3. Add the following environment variables if needed:
   - `PYTHON_VERSION`: 3.8 (or your preferred version)
   - `PORT`: 10000 (or any available port)

4. Click **Create Web Service** to deploy your application.

## Project Structure

```
salary-prediction/
├── app.py                # Main Flask application
├── Procfile             # Render deployment configuration
├── requirements.txt     # Python dependencies
├── static/              # Static files (CSS, JS, images)
│   ├── script.js       # Frontend JavaScript
│   └── style.css       # Styling
├── templates/           # HTML templates
│   └── index.html      # Main application page
└── README.md           # Project documentation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── static/              # Static files (CSS, JS, images)
│   ├── script.js        # Frontend JavaScript
│   └── style.css        # Custom styles
├── templates/           # HTML templates
│   └── index.html       # Main application page
└── adult 3.csv          # Sample dataset
```

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
