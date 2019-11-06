# MLflow Examples

Basic MLflow examples.

## Setup

Do use Python 3.

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

Start the MLflow tracking server.

```
cd $HOME/mlflow-server
source $HOME/virtualenvs/mlflow-examples/bin/activate
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

## Examples
### Setup
Before running an experiment:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Examples
* [hello_world](hello_world) - Hello World
* [sklearn](sklearn) - Scikit learn model
* [pyspark](pyspark) - PySpark model
* [scala_spark](scala_spark) - Scala Spark ML model using the Java client
