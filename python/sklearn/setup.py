from setuptools import setup
  
setup(name='mlflow-sklearn-wine',
      version='0.0.1',
      description='MLflow sklearn wine quality example',
      author='Andre',
      packages=['wine_quality'],
      zip_safe=False,
      python_requires=">=3.7.6",
      install_requires=[
          "mlflow>=1.23.1",
          "scikit-learn==0.24.2",
          "matplotlib==3.2.1",
          "pyarrow>=1.0.0",
          "onnx==1.10.2",
          "onnxmltools==1.10.0",
          "onnxruntime==1.10.0",
          "pyspark==3.2.0",
          "pytest"
    ])
