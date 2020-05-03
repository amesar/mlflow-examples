# mlflow-examples - keras_tf2

## Overview
* Keras with TensorFlow 2.x train and predict.
* Saves model as keras flavor.
* Two experiments:
  * Wine quality dataset.
  * MNIST dataset.
* Option to [autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) parameters and metrics.
* Option to log and score model as ONNX.
* Setup: [conda.yaml](conda.yaml).

## Setup

`conda env create` [conda.yaml](conda.yaml)

## Experiment with Wine Quality Data 

### Training

Source: [wine_train.py](wine_train.py).

```
python wine_train.py --experiment_name keras_wine --epochs 3 --batch_size 128
```


### Scoring

Flavors and formats supported:
* MLflow load.model flavors
  * keras-hd5-model - Keras HD5 format - TensorFlow 2.x legacy
  * keras-hd5-model - pyfunc
  * onnx-model - ONNX flavor
  * onnx-model - pyfunc - does not score correctly unlike Keras with TensorFlow 1.x
* MLflow download.artifact - unsupported flavors
  * tensorflow-model - Standard TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model) format (not HD5 format)
  * tensorflow-lite-model - TensorFlow Lite format

Note: ONNX pyfunc does not score correctly unlike Keras with TensorFlow 1.x.

Source: [wine_predict.py](wine_predict.py).
```
python wine_predict.py --run_id 7e674524514846799310c41f10d6b99d
```

```
mlflow.keras.load_model - runs:/2ff1d36956ab4e479c59db63a0514aaa/keras-hd5-model
model.type: <class 'tensorflow.python.keras.engine.sequential.Sequential'>
predictions.shape: (3428, 1)
+--------------+
|   prediction |
|--------------|
|   -0.753219  |
|   -0.34244   |
+--------------+

mlflow.onnx.load_model - runs:/2ff1d36956ab4e479c59db63a0514aaa/onnx-model
model.type: <class 'onnx.onnx_ONNX_REL_1_6_ml_pb2.ModelProto'>
predictions.shape: (3428, 1)
+--------------+
|   prediction |
|--------------|
|   -0.753219  |
|   -0.34244   |
+--------------+

mlflow.pyfunc.load_model - runs:/2ff1d36956ab4e479c59db63a0514aaa/onnx-model
model.type: <class 'mlflow.onnx._OnnxModelWrapper'>
predictions.shape: (3428, 1)
+--------------+
|   prediction |
|--------------|
|          nan |
|          nan |
+--------------+

keras.models.load_model - tensorflow-model
model.type: <class 'tensorflow.python.keras.saving.saved_model.load.Sequential'>
predictions.shape: (3428, 1)
+--------------+
|   prediction |
|--------------|
|   -0.753219  |
|   -0.34244   |
+--------------+

tf.lite.Interpreter - tensorflow-lite-model - 8040e72452ae4a35b860dc170e42ef8f
model.type: <class 'bytes'>
input_details  [{'name': 'dense_input', 'index': 1, 'shape': array([ 1, 11], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
output_details [{'name': 'Identity', 'index': 0, 'shape': array([1, 1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
predictions.shape: (3428, 1)
+--------------+
|   prediction |
|--------------|
|   -0.753219  |
|   -0.34244   |
+--------------+
```

## Experiment with MNIST Data 

### Training

Source: [mnist_train.py](mnist_train.py).

To run with user logging (no autologging).
```
python mnist_train.py --experiment_name keras_mnist --epochs 3 --batch_size 128
```

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
python mnist_train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --log_as_onnx
```


### Scoring

#### Score as Keras flavor

Source: [mnist_keras_predict.py](mnist_keras_predict.py).
```
python mnist_keras_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model
```

```
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000,)
predictions: [7 2 1 ... 4 5 6]
```

#### Score as Pyfunc flavor

Source: [mnist_pyfunc_predict.py](mnist_pyfunc_predict.py).

##### Score Keras model with Pyfunc 

```
python mnist_pyfunc_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model
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

##### Score ONNX model with Pyfunc 

```
python mnist_pyfunc_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
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


#### Score as ONNX flavor

Source: [mnist_onnx_predict.py](mnist_onnx_predict.py) and [onnx_utils.py](onnx_utils.py).
```
python mnist_onnx_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
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

## Autologging

There are two autologging options:
* keras_autolog - calls mlflow.keras.autolog()
* tensorflow_autolog - calls mlflow.tensorflow.autolog()

Interestingly, they behave differently depending on the TensorFlow version.

| TensorFlow Version | Autolog Method | Params | 
|---|---|---|
| 1x | mlflow.keras.autolog | OK | 
| 1x | mlflow.tensorflow.autolog | none |
| 2x | mlflow.keras.autolog | none | 
| 2x | mlflow.tensorflow.autolog | OK |


```
python mnist_train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --keras_autolog
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
