# mlflow-examples - xgboost

## Overview
* XBGoost with sklearn train and predict with XGBoost.
* Saves model in xgboost format.
* Wine quality dataset [../data/wine-quality-white.csv](../data/wine-quality-white.csv).

## Training

```
python train.py --experiment_name xgboost --estimators 20000 --max_depth 5 
```

## Predictions

You can either use a `runs` or `models` URI.
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/xgboost-model
```

```
python predict.py models:/xgboost_wine/production
```

```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```


