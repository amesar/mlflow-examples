# mlflow-examples - keras

## Overview
* Keras/TensorFlow train and predict.
* Saves model as keras flavor.
* MNIST dataset.
* Option to [autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) parameters and metrics.
* Setup: [conda.yaml](conda.yaml).

## Training

Source: [train.py](train.py).

To run with user logging (no autologging).
```
python main.py --experiment_name keras_mnist --epochs 3 --batch_size 128
```

### Autologging
To run with autologging an no user logging. 
```
python main.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --autolog
```
Autologging will create a model under the name `model`.

Autlogging Parameters:
```
acc
loss
```
Autlogging Metrics:
```
batch_size
class_weight
epochs
epsilon
initial_epoch
learning_rate
max_queue_size
num_layers
optimizer_name
sample_weight
shuffle
steps_per_epoch
use_multiprocessing
validation_freq
validation_split
validation_steps
workers

```

## Predictions

#### Predict as Keras flavor

Source: [keras_predict.py](keras_predict.py).
```
python keras_predict.py runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
predictions: [7 2 1 ... 4 5 6]
```

#### Predict as Pyfunc flavor

Source: [pyfunc_predict.py](pyfunc_predict.py).
```
python pyfunc_predict.py runs:/7e674524514846799310c41f10d6b99d/pyfunc-model
```

```
predictions.type: <class 'pandas.core.frame.DataFrame'>

predictions:                  0             1  ...             8             9
0     7.356894e-07  2.184515e-09  ...  2.648242e-07  1.557131e-05
1     3.679516e-08  5.211977e-06  ...  2.588275e-07  4.540044e-12
...            ...           ...  ...           ...           ...
9998  5.653655e-08  3.749759e-09  ...  1.073899e-04  1.215128e-09
9999  2.790610e-08  2.516971e-11  ...  6.860461e-10  2.355604e-10

```
