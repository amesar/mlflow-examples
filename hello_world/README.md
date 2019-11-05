
# mlflow-spark-summit-2019 - hello_world

Simple Hello World that demonstrates the different ways to run an MLflow experiment.

For details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Synopsis of [hello_world.py](hello_world.py):
* Creates an experiment HelloWorld if it does not exist. 
* Logs parameters, metrics and tags.
* Batch loggging of parameters, metrics and tags.
* No ML training.
* Optionally writes an artifact.

The different ways to run an experiment:
* Unmanaged without mlflow
  * Command-line python
  * Jupyter notebook
* Using mlflow run with [MLproject](MLproject)
  * mlflow run local
  * mlflow run git
  * mlflow run remote

## Setup

**External tracking server**
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**Databricks managed tracking server**
```
export MLFLOW_TRACKING_URI=databricks
```
The token and tracking server URL will be picked up from your Databricks CLI ~/.databrickscfg default profile.

## Running

### Unmanaged without mlflow run
#### Command-line python
```
python hello_world.py
```

#### Jupyter notebook
See [hello_world.ipynb](hello_world.ipynb).
```
export MLFLOW_TRACKING_URI=http://localhost:5000
jupyter notebook
```

### Using mlflow run

#### mlflow run local
```
mlflow run . -Palpha=.01 -Prun_origin=LocalRun -Plog_artifact=True
```
You can also specify an experiment ID:
```
mlflow run . --experiment-id=2019 -Palpha=.01 -Prun_origin=LocalRun -Plog_artifact=True
```

#### mlflow run git
```
mlflow run  https://github.com/amesar/mlflow-fun.git#examples/hello_world \
  --experiment-id=2019 \
  -Palpha=100 -Prun_origin=GitRun -Plog_artifact=True
```
#### mlflow run Databricks remote
Run against Databricks. See [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-databricks) and [cluster.json](cluster.json).
```
mlflow run  https://github.com/amesar/mlflow-fun.git#examples/hello_world \
  --experiment-id=2019 \
  -Palpha=100 -Prun_origin=RemoteRun -Plog_artifact=True \
  -m databricks --cluster-spec cluster.json
```
