# 💼 Salary Prediction System

An intelligent web-based application that predicts employee salary ranges based on demographic and employment details, leveraging the power of **Machine Learning** and a seamless **Flask** web interface.

Designed with a focus on interactivity, automation, and scalability — this project combines clean UI with robust model training capabilities.

---

## 🚀 Features

- 🎯 **Real-time Salary Prediction**  
  Enter employee details and get instant, AI-powered salary range predictions.

- 📊 **Model Training from CSV**  
  Upload your own datasets to train or retrain the model dynamically via the UI.

- 🎨 **Responsive UI/UX**  
  Mobile-friendly layout with smooth visuals and clear feedback for every interaction.

- 📂 **Drag & Drop Upload**  
  User-friendly file upload system that supports drag & drop functionality.

- 📈 **Accuracy Logging & Visual Feedback**  
  Backend logging with real-time model performance insights (e.g., accuracy score).

---

## 🧠 Machine Learning Model

- **Algorithm Used:** Random Forest Classifier  
- **Target Output:** Predict whether an employee earns above or below a salary threshold (classification).  
- **Trained On:**  
  - Age  
  - Work Class  
  - Education  
  - Occupation  
  - Marital Status  
  - Hours per Week

---

## 🛠️ Tech Stack

| Area            | Technologies                       |
|-----------------|------------------------------------|
| Language        | Python 3.10.8                      |
| Framework       | Flask                              |
| ML Model        | Scikit-learn (Random Forest)       |
| Frontend        | HTML5, CSS3, JavaScript            |
| Deployment      | Render (Gunicorn for production)   |
| Dev Tools       | Git, VS Code, Postman              |

---

## 🧪 Local Development

### 🔧 Prerequisites
- Python 3.10.8 (or higher)
- `pip`
- Git

### 📦 Installation Steps

```bash
# Clone the repository
git clone https://github.com/shourya9058/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```
🚀 Running the App Locally
```bash
python app.py
Visit: http://127.0.0.1:5000/
Start predicting or upload a CSV to train a new model.
```
🌐 Deployment (Render)
Connect this repo on Render.

Set up a new Web Service with these settings:

```bash
Build Command:   pip install -r requirements.txt
Start Command:   gunicorn app:app
Python Version:  3.8+ (or your preferred)
Add environment variables if needed (PORT, etc.)

Click "Deploy" and you're live!
```
📁 Project Structure
```bash
Employee-Salary-Prediction/
├── app.py                 # Main Flask app
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment instructions
├── templates/             # HTML (Jinja2 templates)
│   └── index.html
├── static/                # CSS & JS assets
│   ├── style.css
│   └── script.js
├── model.joblib           # Trained ML model
├── adult 3.csv            # Sample dataset
└── README.md              # Project documentation
```
🙌 Contribution Guidelines
Pull requests, feedback, and improvements are welcome.
If you find a bug or have suggestions, feel free to open an issue or PR.

📜 License
This project is licensed under the MIT License.
See LICENSE for more details.

🔗 Useful Links

🔴 Live App: https://employee-salary-prediction-6b7x.onrender.com

🌐 Developer's Portfolio: https://shouryas-portfolio.onrender.com/
