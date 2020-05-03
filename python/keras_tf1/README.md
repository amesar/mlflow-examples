# mlflow-examples - keras_tf1

## Overview
* Keras with TensorFlow 1.x train and predict.
* Saves model as keras flavor.
* MNIST dataset.
* Option to [autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) parameters and metrics.
* Option to log and score model as ONNX.
* Setup: [conda.yaml](conda.yaml).

## Setup

Libraries
```
pip install keras==2.2.5
pip install tensorflow==1.15.0
```

## Training

Source: [train.py](train.py).

To run with user logging (no autologging).
```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128
```

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --log_as_onnx
```

### Autologging

There are two autologging options:
* keras_autolog - calls mlflow.keras.autolog()
* tensorflow_autolog - calls mlflow.tensorflow.autolog()

```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --keras_autolog
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

## Scoring

### Score as Keras flavor

Source: [keras_predict.py](keras_predict.py).
```
python keras_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000,)
predictions: [7 2 1 ... 4 5 6]
```

### Score as Pyfunc flavor

Source: [pyfunc_predict.py](pyfunc_predict.py).

#### Score Keras model with Pyfunc 

```
python pyfunc_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (10000, 10)

predictions:                  0             1  ...             8             9
0     7.356894e-07  2.184515e-09  ...  2.648242e-07  1.557131e-05
1     3.679516e-08  5.211977e-06  ...  2.588275e-07  4.540044e-12
...            ...           ...  ...           ...           ...
9998  5.653655e-08  3.749759e-09  ...  1.073899e-04  1.215128e-09
9999  2.790610e-08  2.516971e-11  ...  6.860461e-10  2.355604e-10
```

#### Score ONNX model with Pyfunc 

```
python pyfunc_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
```
```
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (100000, 1)
predictions:             dense_2
0      7.529760e-07
1      3.086328e-09
...             ...
99998  9.314070e-10
99999  2.785560e-10
```


### Score as ONNX flavor

Source: [onnx_predict.py](onnx_predict.py) and [onnx_utils.py](onnx_utils.py).
```
python onnx_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
```
```
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000, 10)
predictions: 
[[7.5297595e-07 3.0863279e-09 2.5705955e-04 ... 9.9953580e-01 3.5329055e-07 1.3658248e-05]
 [3.7254765e-08 8.8118195e-06 9.9951375e-01 ... 2.6982589e-12 9.4671401e-07 2.8832321e-12]
 ...
 [8.6566203e-08 6.8279524e-09 1.0189680e-08 ... 1.5083194e-09 2.0773137e-04 1.7879515e-09]
 [3.8302844e-08 1.6128991e-11 1.1180904e-05 ... 4.9651490e-12 9.3140695e-10 2.7855604e-10]]
```
