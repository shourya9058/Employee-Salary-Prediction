from setuptools import setup, find_packages

setup(
    name="salary-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask==2.0.3',
        'pandas==1.3.5',
        'numpy==1.21.6',
        'scikit-learn==1.0.2',
        'joblib==1.1.0',
        'matplotlib==3.5.3',
        'seaborn==0.11.2',
        'gunicorn==20.1.0',
        'Werkzeug==2.0.3',
        'scipy==1.7.3',
    ],
    python_requires='>=3.9, <3.10',
    include_package_data=True,
)
