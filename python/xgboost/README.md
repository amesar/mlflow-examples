# mlflow-examples - xgboost

## Overview
* XBGoost with sklearn train and predict.
* Saves model in xgboost format.
* Wine quality dataset [train/wine-quality-white.csv](../../data/train/wine-quality-white.csv).

## Training

```
python train.py --experiment_name xgboost --estimators 20000 --max_depth 5 
```
```
mlflow run . --experiment_name xgboost -P estimators=20000 -P max_depth=5 
```

## Predictions

Score with mlflow.xgboost.load_model and mlflow.pyfunc.load_model.
You can either use a `runs` or `models` URI.
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/xgboost-model
```

```
python predict.py models:/xgboost_wine/production
```

```
=== mlflow.xgboost.load_model
model: <xgboost.core.Booster object at 0x113678b70>
predictions: [5.3752966 5.2566967 5.4596467 ... 5.347645  6.682991  6.0259304]

=== mlflow.pyfunc.load_model
model: <mlflow.xgboost._XGBModelWrapper object at 0x10e9eb198>
predictions: [5.3752966 5.2566967 5.4596467 ... 5.347645  6.682991  6.0259304]
```


