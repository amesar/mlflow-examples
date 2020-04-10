# mlflow-examples - h2o

## Overview
* H2O Random Forest - train and predict
* Saves model in H2O format.
* Wine quality dataset [../data/wine-quality-white.csv](../data/wine-quality-white.csv).

## Setup

```
pip install h2o
```

## Training

```
python train.py --experiment_name h2o --ntrees 5 
```
```
h2o version: 3.28.0.3
Checking whether there is an H2O instance running at http://localhost:54321 ..... not found.
Attempting to start a local H2O server...
. . .

Parse progress: |█████████████████████████████████████████████████████████| 100%
MLflow:
  run_id: 3c552d33dbe145939d60084c662c3af2
  experiment.id: 4
  experiment.name: h2o_wine
  experiment.artifact_location: file:///Users/andre/work/mlflow/server/local_mlrun/mlruns/4
drf Model Build progress: |███████████████████████████████████████████████| 100%
Closing connection _sid_abe4 at exit
H2O session _sid_abe4 closed.
```

## Predictions

Score with `mlflow.h2o.load_model` and `mlflow.pyfunc.load_model`.

###  mlflow.h2o.load_model
```
python h2o_predict.py runs:/7e674524514846799310c41f10d6b99d/h2o-model
```
```
model.type: <class 'h2o.estimators.random_forest.H2ORandomForestEstimator'>
predictions:
        predict
0     6.000000
1     6.000000
2     5.915790
```

###  mlflow.pyfunc.load_model
```
python pyfunc_predict.py runs:/7e674524514846799310c41f10d6b99d/h2o-model
```
```
model: <mlflow.h2o._H2OModelWrapper object at 0x11dff0b90>
predictions:
        predict
0     6.000000
1     6.000000
2     5.915790

```

