# mlflow-examples - keras_tf2 - wine quality


## Overview
* Keras with TensorFlow 2.x train and predict.
* Dataset: Wine quality
* Saves model as MLflow Keras (HD5) flavor.
* Demonstrates advanced features:
  *  Scoring with TensorFlow Serving docker container.
  *  Saves model in different TensorFlow model formats: HD5, SaveModel, TensorFlow Lite and TensorFlow.js.
* Options to:
  * [autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) parameters and metrics.
  * Log and score model as ONNX.

## TensorFlow Serving

TensorFlow model formats:
* [HD5](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format) - Keras TensorFlow 1.x legacy format. This is used for the current MLflow Keras flavor.
* [SavedModel](https://www.tensorflow.org/guide/saved_model) - Standard Keras TensorFlow 2.x Protobuf-based format.
* [TensorFlow Lite](https://www.tensorflow.org/lite) - for mobile and edge devices.
* [TensorFlow.js](https://www.tensorflow.org/js) - for browsers or Node.js.

## Setup

`conda env create` [conda.yaml](conda.yaml)

## TensorFlow Model Serialization Formats

We explore several TensorFlow model formats such as:

* [HD5](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format) - Keras TensorFlow 1.x legacy format. This is used for the current MLflow Keras flavor.
* [SavedModel](https://www.tensorflow.org/guide/saved_model) - Standard Keras TensorFlow 2.x protobuf-based format.
* [TensorFlow Lite](https://www.tensorflow.org/lite) format - for mobile and edge devices.
* [TensorFlow.js](https://www.tensorflow.org/js) - for browsers or Node.js.

MLflow run model details:

* MLflow load.model flavors
  * keras-hd5-model - Keras HD5 format - default for Keras TensorFlow 1.x - `model.h5`
  * keras-hd5-model - pyfunc
  * onnx-model - ONNX flavor
  * onnx-model - pyfunc - does not score correctly unlike Keras TensorFlow 1.x
* MLflow download.artifact - formats not supported as flavors
  * tensorflow-model - Standard TensorFlow SavedModel format - `saved_model.pb`
  * tensorflow-lite-model - TensorFlow Lite
  * tensorflow.js - TensorFlow JS

## Training

Source: [train.py](train.py).

### Options

|Name | Required | Default | Description|
|-----|----------|---------|------------|
| experiment_name | no | none | Experiment name|
| model_name | no | None | Registered model name|
| epochs | no | 5 | Number of epochs |
| batch_size | no | 129 | Batch size |
| log_as_onnx | no | False | Log as ONNX flavor |
| mlflow_custom_log | no | True | Log params/metrics with mlflow.log |
| keras_autolog | no | False | Automatically log params/ metrics with mlflow.keras.autolog |
| tensorflow_autolog | no | False | Automatically log params/ metrics with mlflow.tensorflow.autolog |


### Run
```
python train.py --experiment_name keras_wine --epochs 3 --batch_size 128
```

## Batch Scoring

Note: ONNX pyfunc does not score correctly unlike Keras with TensorFlow 1.x.

Source: [predict.py](predict.py).

### Options

|Name | Required | Default | Description|
|-----|----------|---------|------------|
| run_id | yes | none | run_id |
| score_as_pyfunc | no | False | Score as PyFunc  |

### Run
```
python predict.py --run_id 7e674524514846799310c41f10d6b99d
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

## Real-time Scoring - MLflow


### Data
[../../data/predict-wine-quality.json](../../data/predict-wine-quality.json)

### Web server

### Docker container

## Real-time Scoring - TensorFlow Serving

### Data

[../../data/tensorflow_serving.json](../../data/tensorflow_serving.json)
```
{"instances": [ 
  [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ],
  [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ],
  [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ]
] }
```

### Launch scoring server as docker container

```
docker run -t --rm --publish 8501 \
--volume /opt/mlflow/mlruns/1/f48dafd70be044298f71488b0ae10df4/artifacts/tensorflow-model:/models/keras_wine\
--env MODEL_NAME=keras_wine \
tensorflow/serving
```


### Score 

```
curl -d '{"instances": [12.8, 0.03, 0.48, 0.98, 6.2, 29, 1.2, 0.4, 75 ] }' \
     -X POST http://localhost:8501/v1/models/keras_wine:predict
```

```
{ "predictions": [[-0.70597136]] }
```
