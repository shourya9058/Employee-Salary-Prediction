from setuptools import setup, find_packages

setup(
    name="salary-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.3',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'gunicorn>=20.1.0',
    ],
    python_requires='>=3.10, <3.11',
    include_package_data=True,
)
