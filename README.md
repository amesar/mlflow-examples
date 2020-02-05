# MLflow Examples

MLflow examples - basic and advanced.

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

### Docker Tracking Server

See [docker/README](docker/README.md).

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

#### Database-backed store

##### MySQL Setup
* Install MySQL
* Create an mlflow user with password.
* Create a database `mlflow` 

##### Start the MLflow Tracking Server
```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri mysql://MLFLOW_USER:MLFLOW_PASSWORD@localhost:3306/mlflow \
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

### Samples

**Python examples**
* [hello_world](python/hello_world) - Hello World - no training or scoring.
* [sklearn](python/sklearn) - Scikit-learn model - train and score. ONNX too.
* [sparkml](python/sparkml) - Spark ML model - train and score.  ONNX too.
* [keras](python/keras) - Keras/Tensorflow - train and score. ONNX too.
* [xgboost](python/xgboost) - XGBoost (sklearn wrapper) model - train and score.
* [pytorch](python/pytorch) - Pytorch  - train and score. ONNX too.
* [onnx](python/onnx) - Convert sklearn model to ONNX flavor - train and score.
* [model_registry](python/model_registry) - Jupyter notebook sampling the Model Registry API.
* [reproduce](python/reproduce) - Reproduce an existing run.
* [benchmarks](python/benchmarks) - Simple scoring server performance benchmark.

The sklearn and sparkml examples also showcase:
* Different ways to run a project with the mlflow CLI 
* Real-time server scoring with docker containers
* Running a project against a Databricks cluster

**Scala examples - uses the MLflow Java client**
* [hello_world](scala/sparkml/README.md#hello_world) - Hello World - no training or scoring.
* [sparkml](scala/sparkml/) - Scala Spark ML train and score.
* [mleap](scala/mleap) - Score an MLeap model with MLeap runtime (no Spark dependencies).

**Other**
* [docker](docker) - MLflow tracking server and MySQL database containers.

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

