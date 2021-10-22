from setuptools import setup
  
setup(name='mlflow-sklearn-wine',
      version='0.0.1',
      description='MLflow sklearn wine quality example',
      author='Andre',
      packages=['wine_quality'],
      zip_safe=False,
      python_requires=">=3.7.6",
      install_requires=[
          "mlflow>=1.17.0",
          "scikit-learn==0.24.2",
          "matplotlib==3.2.1",
          "pyarrow>=1.0.0",
          "onnx==1.10.1",
          "onnxmltools==1.9.1",
          "onnxruntime==1.8.1",
          "pyspark==3.1.2",
          "pytest"
    ])
