from setuptools import setup, find_packages

setup(
    name="salary-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask==2.0.3',
        'pandas==1.5.3',
        'numpy==1.23.5',
        'scikit-learn==1.2.2',
        'joblib==1.2.0',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'gunicorn==20.1.0',
        'Werkzeug==2.0.3',
        'scipy==1.10.1',
    ],
    python_requires='>=3.10, <3.11',
    include_package_data=True,
)
