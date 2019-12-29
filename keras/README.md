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
python main.py --experiment_name sklearn --epochs 3 --batch_size 128
```

### Autologging
To run with autologging an no user logging. 
```
python main.py --experiment_name sklearn --epochs 3 --batch_size 128 --autolog
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

Source: [predict.py](predict.py).
You can either use a `runs` or `models` URI.
```
python predict.py runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
python predict.py models:/keras-wine/production
```

```
predictions: [7 2 1 ... 4 5 6]
```
