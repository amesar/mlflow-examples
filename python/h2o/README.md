# mlflow-examples - h2o

## Overview
* Train and predict with H2O
* Two training programs:
  * H2O Random Forest - simple train 
  * H2O AutoML
* Saves model in H2O format.
* Wine quality dataset [../data/wine-quality-white.csv](../data/wine-quality-white.csv).

## Setup

```
conda env create --file conda.yaml
conda activate mlflow-examples-h2o
```

## Train

### Simple Train

Source: [train.py](train.py).

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

### AutoML Train

Source: [automl_train.py](automl_train.py).

```
python automl_train.py --experiment_name h2o_automl --ntrees 5 
```

```
model_id                                               mean_residual_deviance      rmse       mse       mae      rmsle    training_time_ms    predict_time_per_row_ms
---------------------------------------------------  ------------------------  --------  --------  --------  ---------  ------------------  -------------------------
StackedEnsemble_BestOfFamily_AutoML_20200412_231704                  0.392527  0.62652   0.392527  0.449962  0.09409                   634                   0.015658
StackedEnsemble_AllModels_AutoML_20200412_231704                     0.392923  0.626836  0.392923  0.449871  0.0941326                 706                   0.036654
DRF_1_AutoML_20200412_231704                                         0.399265  0.631874  0.399265  0.45661   0.0951139                 828                   0.00458
XGBoost_2_AutoML_20200412_231704                                     0.405155  0.636518  0.405155  0.464714  0.0954654                3969                   0.005445
XGBoost_1_AutoML_20200412_231704                                     0.415968  0.644956  0.415968  0.474978  0.0965878                3268                   0.003939
GBM_4_AutoML_20200412_231704                                         0.431667  0.657014  0.431667  0.493813  0.0984208                 240                   0.003878
GBM_3_AutoML_20200412_231704                                         0.436481  0.660667  0.436481  0.500997  0.0988277                 250                   0.004338
GBM_1_AutoML_20200412_231704                                         0.441423  0.664397  0.441423  0.507534  0.0994319                 308                   0.004561
GBM_2_AutoML_20200412_231704                                         0.443043  0.665615  0.443043  0.510294  0.0994141                 214                   0.007207
XGBoost_3_AutoML_20200412_231704                                     0.451612  0.672021  0.451612  0.520061  0.100172                 1385                   0.002573


```

## Predict

Score with `mlflow.h2o.load_model` and `mlflow.pyfunc.load_model`.

###  mlflow.h2o.load_model

Source: [h2o_predict.py](h2o_predict.py).

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
Source: [pyfunc_predict.py](pyfunc_predict.py).
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
