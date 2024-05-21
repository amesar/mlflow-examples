from setuptools import setup
  
setup(name='mlflow-sklearn-wine',
      version='0.0.1',
      description='MLflow sklearn wine quality advanced example',
      author='Andre',
      packages=['wine_quality'],
      zip_safe=False,
      python_requires="==3.8.13",
      install_requires=[
          "mlflow>=2.13.0",
          "scikit-learn",
          "matplotlib",
          "onnx==1.15.0",
          "onnxruntime==1.16.3",
          "skl2onnx==1.16.0",
          "pyspark==3.3.1",
          "shortuuid",
          "pytest",
          "pytest-ordering"
    ])
