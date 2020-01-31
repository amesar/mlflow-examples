# mlflow-examples - reproducibility

## Overview
Tool to reproduce a run with the [mlflow.projects.run()](https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run) (equivalent to [mlflow run](https://mlflow.org/docs/latest/projects.html#running-projects) CLI command).

MLflow allows you to associate the git repository URI and commit hash for a run's source.
Using these two attributes we leverage MLflow's `project` feature (API `projects.run` or CLI `mlflow run`).

[reproduce_run.py](reproduce_run.py) can reproduce any run in a generic fashion. Note that you should record any data files as a parameter.

Logic:
* Specify the run_id of the reference run you wish to reproduce
* Execute a new run with `projects.run()` whose arguments are from the parameters of the reference run
* Compare that metrics for the two runs are the same

### Arguments

|Name | Required | Default | Description|
|---|---|---|---|
| uri | yes | | URI to github repo |
| run_id | yes | | Run ID  |
| experiment_name | no | Default | Name of new reproduced experiment where the run will live|
| verbose | no | False | Display run details  |

### Reproducible Run
```
python reproduce_run.py \
  --uri https://github.com/amesar/mlflow-examples.git#python/sklearn \
  --run_id d1b4b10969174863a2bad1342f6746ce 
```
```
+----------+----------+----------+
| Metric   |     Run1 |     Run2 |
|----------+----------+----------|
| mae      | 0.587477 | 0.587477 |
| r2       | 0.283681 | 0.283681 |
| rmse     | 0.749066 | 0.749066 |
+----------+----------+----------+
Runs equal: True
```

### Not Reproducible Run

For example, a Keras run may not be 100% reproducible due to non-deterministic initialization of weights.
```
python reproduce_run.py \
  --uri https://github.com/amesar/mlflow-examples.git#python/keras \
  --run_id 5500ff3b12f76aeb500120edfa10d0082c87bad0
```
```
+-----------+-----------+-----------+
| Metric    |      Run1 |      Run2 |
|-----------+-----------+-----------|
| test_acc  | 0.9733    | 0.9733    |
| test_loss | 0.0885112 | 0.0897533 |
+-----------+-----------+-----------+
Runs equal: False
```
