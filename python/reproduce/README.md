# mlflow-examples - reproduce

Fully automated push-button run reproducibility.

## Overview
The MLflow [Project](https://mlflow.org/docs/latest/projects.html) concept allows for reproducibility of experiment runs.
When you run `mlflow run` the resulting run's git repository path and commit hash are saved in the `mlflow.source.name` and `mlflow.source.git.commit` tags. Then, at a later date you can re-run the code by leveraging these two tags.

Currently the entry point to `mlflow run` is a source code path (git URI or local path) and not a run ID.
The [run_reproducer.py](run_reproducer.py) script is a tool that enables you to reproduce a run by run ID.

Think of run_reproducer as doing the following:
`  mlflow run runs:/bd341538c53a452bbbb7e204a6430a25`

**Example `mlflow run`**

MLFlow [system tags](https://mlflow.org/docs/latest/tracking.html#system-tags) for the [sklearn](../sklearn) example:
```
mlflow.gitRepoURL: https://github.com/amesar/mlflow-examples.git
mlflow.project.backend: local
mlflow.project.entryPoint: main
mlflow.project.env: conda
mlflow.runName: GitRun
mlflow.source.git.commit: 665abdbc860341b70f467a28255ecbfd05f6b121
mlflow.source.git.repoURL: https://github.com/amesar/mlflow-examples.git
mlflow.source.name: https://github.com/amesar/mlflow-examples.git#python/sklearn
mlflow.source.type: PROJECT
mlflow.user: andre
mlflow_version: 1.6.0
```

Using the `mlflow.source.name` and `mlflow.source.git.commit` tags, we can then reproduce the run:
```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  --version=665abdbc860341b70f467a28255ecbfd05f6b121 \
  --experiment-name=sklearn_mlflow_cli \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=gitRun/repro 
```


## Running `run_reproducer`

### Overview

Logic:
* Specify the run_id of the reference run you wish to reproduce
* Execute a new run with [mlflow.projects.run](https://mlflow.org/docs/latest/python_api/mlflow.projects.html#mlflow.projects.run) whose arguments are set from the parameters of the reference run
* Compare that metrics for the two runs are the same

### Arguments

|Name | Required | Default | Description|
|---|---|---|---|
| run_id | yes | | Run ID  |
| experiment_name | no | Default | Name of experiment under which to launch the run |
| rel_tol | no | 1e-09 | Relative tolerance for metrics comparison with [math.isclose](https://docs.python.org/3/whatsnew/3.5.html#pep-485-a-function-for-testing-approximate-equality) |
| verbose | no | False | Display run details  |

### Reproducible Run
```
python run_reproducer.py \
  --run_id bd341538c53a452bbbb7e204a6430a25 \
  --experiment_name repo_scratch \
  --rel_tol 1e-02 
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

Note that the meaning of reproducible is algorithm-specific. 
For example, a Keras run may not be 100% reproducible due to the non-deterministic initialization of weights.
In this case, you may wish to fiddle with the `rel_tol` argument or provide your own equality function.
```
+-----------+-----------+-----------+
| Metric    |      Run1 |      Run2 |
|-----------+-----------+-----------|
| test_acc  | 0.9733    | 0.9733    |
| test_loss | 0.0885112 | 0.0897533 |
+-----------+-----------+-----------+
Runs equal: False
```

