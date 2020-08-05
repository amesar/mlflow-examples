# mlflow-examples - Keras/TensorFlow - Wine Quality


## Overview

* Keras TensorFlow 2.x 
* Train and predict with a number of model flavors and formats
* Algorithm: KerasRegressor
* Dataset: Wine quality
* Model flavors and formats
  * Logs model as MLflow Keras (HD5) flavor 
  * Saves model in a number of other TensorFlow formats (non-flavors) such as SavedModel, TensorFlow Lite and TensorFlow.js
  * Logs and scores model as ONNX flavor.
* Real-time scoring
 * Launches an MLflow scoring server either as a local web server or a docker container.
 * Launches a [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) docker container
* Autolog parameters and metrics - [keras.autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) - [tensorflow.autolog](https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#mlflow.tensorflow.autolog).

## Setup

`conda env create` [conda.yaml](conda.yaml)

`conda activate mlflow-examples-keras_tf_wine`

## Training

Source: [train.py](train.py).

### Options

|Name | Required | Default | Description|
|-----|----------|---------|------------|
| experiment_name | no | none | Experiment name|
| model_name | no | None | Registered model name|
| epochs | no | 5 | Number of epochs |
| batch_size | no | 128 | Batch size |
| mlflow_custom_log | no | True | Explicitly log params and metrics with mlflow.log |
| keras_autolog | no | False | Automatically log params and metrics with mlflow.keras.autolog |
| tensorflow_autolog | no | False | Automatically log params and metrics with mlflow.tensorflow.autolog |
| log_as_onnx | no | False | Log as ONNX flavor |
| log_as_tensorflow_lite | no | False | Log as TensorFlow Lite |
| log_as_tensorflow_js | no | False | Log as TensorFlow JS |


### Run
```
mlflow run -P experiment_name=keras_wine -P epochs3 -P batch_size=128
```
or
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
| score_as_pyfunc | no | True | Score as PyFunc  |
| score_as_tensorflow_lite | no | False | Score as TensorFlow Lite  |

### Run
```
python predict.py --run_id 7e674524514846799310c41f10d6b99d --score_as_tensorflow_lite True
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

Launch the MLflow scoring server either as a local web server or inside a docker container.

### Data
[../../data/score/wine-quality.json](../../data/score/wine-quality.json)

### Web server

Launch the web server.
```
mlflow pyfunc serve -port 5001 \
  -model-uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model
```

### Docker container

You can use the SageMaker container on your local machine without SageMaker dependencies.

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-keras
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model \
  --port 5001 --image sm-wine-keras
```

### Score 
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/score/wine-quality.json \
  http://localhost:5001/invocations
```
```
[
  [5.470588235294118,5.470588235294118,5.769607843137255]
]
```


## Real-time Scoring - TensorFlow Serving

A number of TensorFlow model formats are saved with the run.
Note that `keras-hd5-model` is the only official MLflow flavor. Other model formats are mere artifacts.

* keras-hd5-model - [HD5](https://www.tensorflow.org/tutorials/keras/save_and_load#hdf5_format). Keras TensorFlow 1.x legacy format. This is used for the current MLflow Keras flavor.
* tensorflow-model - TensorFlow [SavedModel](https://www.tensorflow.org/guide/saved_model). Standard Keras TensorFlow 2.x Protobuf-based format.
* tensorflow-lite - [TensorFlow Lite](https://www.tensorflow.org/lite). For mobile and edge devices.
* tensorflow-js - [TensorFlow.js](https://www.tensorflow.org/js). For browsers or Node.js.

### Launch TensorFlow scoring server as docker container

See [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).

```
docker run -t --rm --publish 8502 \
--volume /opt/mlflow/mlruns/1/f48dafd70be044298f71488b0ae10df4/artifacts/tensorflow-model:/models/keras_wine\
--env MODEL_NAME=keras_wine \
tensorflow/serving
```

### Score 

```
curl http://localhost:8502/v1/models/keras_wine:predict -X POST \
  -d '{"instances":  [[ 7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8 ]] }' 
```
```
{ "predictions": [[-0.70597136]] }
```
or
```
curl http://localhost:8502/v1/models/keras_wine:predict -X POST \
  -d @../../data/score/tf-serving-wine-quality.json 
```
```
{
    "predictions": [[-0.70597136], [-0.518047869], [-0.380209982] ]
}
```

### Data

[../../data/score/tf-serving-wine-quality.json](../../data/score/tf-serving-wine-quality.json)
```
{"instances": [ 
  [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ],
  [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ],
  [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ]
] }
```
