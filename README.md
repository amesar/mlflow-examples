# MLflow Examples

Basic MLflow examples.

## Setup

Basic.
* Use Python 3
* Install Spark on your machine

Create a virtual environment.
```
cd $HOME/virtualenvs
python -m venv mlflow-examples
```

Install libraries.
```
source $HOME/virtualenvs/mlflow-examples/bin/activate
pip install mlflow==1.4.0
pip install sklearn
pip install matplotlib
pip install pyarrow
```

## MLflow Server

You can either use the local file store or a database-backed store. 
See MLflow [Storage](https://mlflow.org/docs/latest/tracking.html#storage) documentation.

Note that apparently new MLflow 1.4.0 Model Registry functionality is only available with the database-backed store.

First activate the virtual environment.
```
cd $HOME/mlflow-server
source $HOME/virtualenvs/mlflow-examples/bin/activate
```


### File Store

Start the MLflow tracking server.

```
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

### Database-backed store

#### MySQL Setup
* Install MySQL
* Create an mlflow user with password.
* Create a database `mlflow` 

#### Start the MLflow Tracking Server
```
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri mysql://MLFLOW_USER:MLFLOW_PASSWORD@localhost:3306/mlflow \
  --default-artifact-root $PWD/mlruns  
```

## Examples
### Setup
Before running an experiment
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Examples
* Python examples
  * [hello_world](hello_world) - Hello World
  * [sklearn](sklearn) - Scikit learn model
  * [pyspark](pyspark) - PySpark model
* [scala_spark](scala_spark) - Scala Spark ML model using the MLflow Java client
