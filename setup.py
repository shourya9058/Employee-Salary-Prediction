from setuptools import setup, find_packages

setup(
    name="salary-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.0.3',
        'pandas==1.5.3',
        'numpy==1.24.3',
        'scikit-learn==1.2.2',
        'joblib==1.2.0',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'gunicorn==20.1.0',
        'Werkzeug==2.0.3',
    ],
    python_requires='>=3.8,<3.11',
)
