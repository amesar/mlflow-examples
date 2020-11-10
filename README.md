# MLflow Examples

MLflow examples - basic and advanced.

## Examples

**Python examples**
* [hello_world](python/hello_world) - Hello World - no training or scoring.
* [sklearn](python/sklearn) - Scikit-learn model - train and score. 
  * Canonical example that shows multiple ways to train and score.
  * Options to log ONNX model, autolog and save model signature.
  * Train locally or against a Databricks cluster.
  * Score real-time against a local web server or Docker container.
  * Score batch with mlflow.load_model or Spark UDF>
* [sparkml](python/sparkml) - Spark ML model - train and score. ONNX too.
* Keras/Tensorflow - train and score. ONNX working too.
  * Keras with TensorFlow 2.x
    * [keras_tf_wine](python/keras_tf_wine) - Wine quality dataset
    * [keras_tf_mnist](python/keras_tf_mnist) - MNIST dataset
  * [keras_tf1](python/keras_tf1) - Keras with TensorFlow 1.x - legacy
* [xgboost](python/xgboost) - XGBoost (sklearn wrapper) model - train and score.
* [catboost](python/catboost) - Catboost (using sklearn) model - train and score. ONNX working too.
* [pytorch](python/pytorch) - Pytorch  - train and score. ONNX too.
* [onnx](python/onnx) - Convert sklearn model to ONNX flavor - train and score.
* [h2o](python/h2o) - H2O model - train and score - with AutoML. ONNX too.
* [model_registry](python/model_registry) - Jupyter notebook sampling the Model Registry API.
* [e2e-ml-pipeline](python/e2e-ml-pipeline) - End-to-end ML pipeline - training to real-time scoring.
* [reproduce](python/reproduce) - Reproduce an existing run.
* [scoring_server_benchmarks](python/scoring_server_benchmarks) - Scoring server performance benchmarks.

The sklearn and Spark ML examples also demonstrate:
* Different ways to run a project with the mlflow CLI 
* Real-time server scoring with docker containers
* Running a project against a Databricks cluster

**Scala examples - uses the MLflow Java client**
* [hello_world](scala/sparkml/README.md#hello_world) - Hello World - no training or scoring.
* [sparkml](scala/sparkml/) - Scala train and score - Spark ML and XGBoost4j
* [mleap](scala/mleap) - Score an MLeap model with MLeap runtime (no Spark dependencies).
* [onnx](scala/onnx) - Score an ONNX model (that was created in Scikit-learn) in Java.

**Databricks**
* [databricks_notebooks](databricks_notebooks) - Databricks notebooks
* [Notebook CICD](databricks_notebooks/cicd) - Lighweight CICD example with Databricks notebook

**Docker**
* [docker/docker-server](docker/docker-server) - MLflow tracking server and MySQL database containers.

## Setup

Use Python 3.7.5

* For Python environment use either:
  * Miniconda with [conda.yaml](python/conda.yaml).
  * Virtual environment with PyPi.
* Install Spark 2.4.2 on your machine.
* For ONNX examples also install:
  * onnx==1.6.0
  * onnxmltools==1.6.0
  * skl2onnx==1.6.0
  * onnxruntime==1.1.0

### Miniconda

* Install miniconda3: ``https://conda.io/miniconda.html``
* Create the environment: ``conda env create --file conda.yaml``
* Source the environment: `` source activate mlflow-examples``

### Virtual Environment

Create a virtual environment.
```
python -m venv mlflow-examples
source mlflow-examples/bin/activate
```

`pip install` the libraries in conda.yaml.

## MLflow Server

You can either run the MLflow tracking server directly on your laptop or with Docker.

### Docker 

See [docker/docker-server/README](docker/docker-server/README.md).

### Laptop Tracking Server

You can either use the local file store or a database-backed store. 
See MLflow [Storage](https://mlflow.org/docs/latest/tracking.html#storage) documentation.

Note that new MLflow 1.4.0 Model Registry functionality seems only to work with the database-backed store.

First activate the virtual environment.
```
cd $HOME/mlflow-server
source $HOME/virtualenvs/mlflow-examples/bin/activate
```


#### File Store

Start the MLflow tracking server.

```
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

#### Database-backed store - MySQL

* Install MySQL
* Create an mlflow user with password.
* Create a database `mlflow` 

Start the MLflow Tracking Server
```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri mysql://MLFLOW_USER:MLFLOW_PASSWORD@localhost:3306/mlflow \
  --default-artifact-root $PWD/mlruns  
```

#### Database-backed store - SQLite

```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root $PWD/mlruns  
```

## Examples

Most of the examples use a DecisionTreeRegressor model with the wine quality data set.

As such, the `python/sparkml` and `scala/sparkml` are isomorphic as they are simply language variants of the same Spark ML algorithm.

### Setup
Before running an experiment
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```


## Data

Data is in the [data](data) folder.

[wine-quality-white.csv](data/wine-quality-white.csv) contains the training data.

Real-time scoring prediction data
* The prediction files contain the first three records of wine-quality-white.csv. 
* The format is standard MLflow JSON-serialized Pandas DataFrames split orientation format described [here](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models).
* Data in [predict-wine-quality.json](data/predict-wine-quality.json) is directly derived from wine-quality-white.csv.
  * The values are a mix of integers and doubles.
* Apparently if you score predict-wine-quality.json against an MLeap SageMaker container, you will get errors as the server is unable to handle integers (bug).
* Hence [predict-wine-quality-float.json](data/predict-wine-quality-float.json) whose data is all doubles.

