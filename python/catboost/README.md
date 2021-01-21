# mlflow-examples - catboost

## Overview
* CatBoost using sklearn flavor for train and predict.
* Saves model in CatBoost and ONNX format.
* Wine quality dataset [../../data/wine-quality-white.csv](../../data/wine-quality-white.csv).

## Training

```
python train.py --experiment_name catboost --iterations 10000 --depth 5 --learning_rate 1
```

```
mlflow run . --experiment-name=catboost -P iterations=10000 -P depth=5 -P learning_rate=1
```

## Sklearn and Pyfunc Predictions

Score with `mlflow.sklearn.load_model` and `mlflow.pyfunc.load_model`.
You can either use a `runs` or `models` URI.
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/catboost-model
```

```
python predict.py models:/catboost/production
```

```
mlflow.catboost.load_model
model: <class 'catboost.core.CatBoostRegressor'>
predictions: [5.3752966 5.2566967 5.4596467 ... 5.347645  6.682991  6.0259304]

mlflow.pyfunc.load_model
model: <class 'catboost.core.CatBoostRegressor'>
predictions: [5.3752966 5.2566967 5.4596467 ... 5.347645  6.682991  6.0259304]
```

## ONNX Prediction

The ONNX scorer is very slow. Whereas [predict.py](predict.py) takes 2 seconds, [onnx_predict.py](onnx_predict.py) takes 35 seconds!
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/onnx-model
```
```
model: <class 'onnx.onnx_ONNX_REL_1_6_ml_pb2.ModelProto'>
predictions: [[5.999989 ]
 [5.9954762]
 [6.3790827]
 ...
 [5.9997087]]
```
