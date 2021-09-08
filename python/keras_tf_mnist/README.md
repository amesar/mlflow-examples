# mlflow-examples - Keras/TensorFlow - MNIST


## Overview
* Keras with TensorFlow 2.x train and predict.
* Dataset: MNIST dataset.
* Demonstrates how to serve model with:
  * MLflow scoring server
  * TensorFlow Serving
* Saves model as:
  *  MLflow Keras HD5 flavor 
  *  TensorFlow 2.0 SavedModel format as artifact for TensorFlow Serving
* Options to:
  * [autolog](https://mlflow.org/docs/latest/python_api/mlflow.keras.html#mlflow.keras.autolog) parameters and metrics.
  * log and score model as ONNX.

## Setup

`conda env create` [conda.yaml](conda.yaml)

`conda activate mlflow-examples-keras_tf_mnist`


## Training

Source: [train.py](train.py).

### Options
```
  --experiment-name TEXT        Experiment name
  --model-name TEXT             Registered model name
  --data-path TEXT              Data path
  --epochs INTEGER              Epochs
  --batch-size INTEGER          Batch size
  --repeats INTEGER             Repeats
  --mlflow-custom-log BOOLEAN   Explicitly log params, metrics and model with mlflow.log_
  --keras-autolog BOOLEAN       Automatically log params, metrics and model with mlflow.keras.autolog
  --tensorflow-autolog BOOLEAN  Automatically log params, metrics and model with mlflow.tensorflow.autolog
  --mlflow-autolog BOOLEAN      Automatically log params, metrics and model with mlflow.autolog
  --log-as-onnx BOOLEAN         Log an ONNX model also
```

### Run training

```
python train.py --experiment-name keras_mnist --epochs 3 --batch-size 128
```
or
```
mlflow run . --experiment-name keras_mnist -P epochs=3 -P batch-size=128
```

### SavedModel format

The training program saves the Keras model in two formats:

  * HD5 format - MLflow saves Keras model by default in the legacy HD5 format. Logged as `keras-model-h5` artifact.
  * [SavedModel](https://www.tensorflow.org/guide/saved_model) - current TensorFlow preferred model format. Logged as `keras-model-tf` artifact.


### ONNX

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
mlflow run . --experiment-name keras_mnist -P epochs=3 -P batch-size=128 -P log-as-onnx=True
```

ONNX training works with TensorFlow 2.3.0 but fails on 2.4.0 with the following message:
```
 File "/home/mlflow-examples/python/keras_tf_mnist/onnx_utils.py", line 7, in log_model
    onnx_model = onnxmltools.convert_keras(model, artifact_path)
  File "/opt/conda/lib/python3.7/site-packages/onnxmltools/convert/main.py", line 33, in convert_keras
    return convert(model, name, doc_string, target_opset, channel_first_inputs)
  File "/opt/conda/lib/python3.7/site-packages/keras2onnx/main.py", line 62, in convert_keras
    tf_graph = build_layer_output_from_model(model, output_dict, input_names, output_names)
  File "/opt/conda/lib/python3.7/site-packages/keras2onnx/_parser_tf.py", line 304, in build_layer_output_from_model
    graph = model.outputs[0].graph
AttributeError: 'KerasTensor' object has no attribute 'graph'
```
However, it works when run on a Linux-based container.

## Batch Scoring

### Data

By default the prediction scripts get their data from `tensorflow.keras.datasets.mnist.load_data()`.
To specify another file, use the `data_path` option. 
See get_prediction_data() in [utils.py](utils.py) for details.

The following formats are supported:

* json - standard MLflow [JSON-serialized pandas DataFrames](https://mlflow.org/docs/latest/models.html#local-model-deployment) format.
See example [mnist-mlflow.json](../../data/score/mnist/mnist-mlflow.json).
* csv - CSV version of above. See example [mnist-mlflow.csv](../../data/score/mnist/mnist-mlflow.csv).
* npz - Compressed Numpy format.
* png - Raw PNG image.


### Score as Keras and PyFunc flavor 

Score as Keras and Keras/PyFunc flavor.
Source: [keras_predict.py](keras_predict.py).

```
python keras_predict.py --model-uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
**** mlflow.keras.load_model

model.type: <class 'tensorflow.python.keras.engine.sequential.Sequential'>
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000, 10)
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|           0 |           1 |           2 |           3 |           4 |           5 |           6 |           7 |           8 |           9 |
|-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------|
| 3.123e-06   | 2.60792e-07 | 0.000399815 | 0.000576044 | 3.31058e-08 | 1.12318e-05 | 1.5746e-09  | 0.998986    | 9.80188e-06 | 1.36477e-05 |
| 1.27407e-06 | 5.95377e-05 | 0.999922    | 3.0263e-06  | 6.65168e-13 | 6.7665e-06  | 6.27953e-06 | 1.63278e-11 | 1.39965e-06 | 4.86269e-12 |
.  . .
| 4.17418e-07 | 6.36174e-09 | 8.52869e-07 | 1.0931e-05  | 0.0288905   | 2.07351e-06 | 6.78868e-08 | 0.000951144 | 0.00079286  | 0.969351    |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+

**** mlflow.pyfunc.load_model

model.type: <class 'mlflow.pyfunc.PyFuncModel'>
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (10000, 10)
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
|           0 |           1 |           2 |           3 |           4 |           5 |           6 |           7 |           8 |           9 |
|-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------|
| 3.123e-06   | 2.60792e-07 | 0.000399815 | 0.000576044 | 3.31058e-08 | 1.12318e-05 | 1.5746e-09  | 0.998986    | 9.80188e-06 | 1.36477e-05 |
| 1.27407e-06 | 5.95377e-05 | 0.999922    | 3.0263e-06  | 6.65168e-13 | 6.7665e-06  | 6.27953e-06 | 1.63278e-11 | 1.39965e-06 | 4.86269e-12 |
.  . .
| 4.17418e-07 | 6.36174e-09 | 8.52869e-07 | 1.0931e-05  | 0.0288905   | 2.07351e-06 | 6.78868e-08 | 0.000951144 | 0.00079286  | 0.969351    |
+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+-------------+
```

### Score as Pyfunc flavor

Source: [pyfunc_predict.py](pyfunc_predict.py).

#### Score Keras model with Pyfunc 

```
python pyfunc_predict.py --model-uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (9, 10)
predictions:               0             1  ...             8             9
0  1.274071e-06  5.953768e-05  ...  1.399654e-06  4.862691e-12
1  4.793732e-06  9.923353e-01  ...  1.608626e-03  1.575307e-04
8  4.174167e-07  6.361756e-09  ...  7.928589e-04  9.693512e-01
```

#### Score ONNX model with Pyfunc 

```
python pyfunc_predict.py --model-uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
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
python onnx_predict.py --model-uri runs:/7e674524514846799310c41f10d6b99d/onnx-model
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

## Real-time Scoring

Two real-time scoring server solutions are shown here:
* MLflow scoring server
* TensorFlow Servering scoring server

### Real-time Scoring Data

Scoring request data is generated from the reshaped MNIST data saved as JSON data per each scoring server's format.
For details see [create_scoring_datafiles.py](create_scoring_datafiles.py).
* MLflow scoring server - [../../data/score/mnist/mnist-mlflow.json](../../data/score/mnist/mnist-mlflow.json)
* TensorFlow Serving - [../../data/score/mnist/mnist-tf-serving.json](../../data/score/mnist/mnist-tf-serving.json).


### Real-time Scoring - MLflow

You can launch the the MLflow scoring server in the following ways:
* Local web server 
* Docker container

Data: [../../data/score/mnist/mnist-mlflow.json](../../data/score/mnist/mnist-mlflow.json)

#### 1. Local Web server

```
mlflow pyfunc serve -port 5001 \
  -model-uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

#### 2. Docker container

You can run MLflow's SageMaker container on your local machine.

See MLflow documentation: [Deploy a python_function model on Amazon SageMaker](https://www.mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-amazon-sagemaker).

```
mlflow sagemaker build-and-push-container --build --no-push --container sm-mnist-keras
```

```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/keras-model \
  --port 5001 --image sm-mnist-keras
```

#### Score

##### Score JSON file.
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/score/mnist/mnist-mlflow.json \
  http://localhost:5001/invocations
```
```
[
  {
    "0": 3.122993575743749e-06,
    "1": 2.6079169401782565e-07,
    "2": 0.0003998146567028016,
    "3": 0.0005760430940426886,
    "4": 3.3105706620517594e-08,
    "5": 1.1231797543587163e-05,
    "6": 1.5745946768674912e-09,
    "7": 0.9989859461784363,
    "8": 9.801864507608116e-06,
    "9": 1.3647688319906592e-05
  },
. . .
]
```

##### Score MNIST PNG file.
```
python convert_png_to_mlflow_json.py ../../data/score/mnist/0_9993.png | \
curl -X POST -H "Content-Type:application/json" \
  -d @- \
  http://localhost:5001/invocations
```
```
[
  {
    "0": 1,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0
  }
]
```

### TensorFlow Serving Real-time Scoring

Overview:

* Serve an MLflow Keras/TensorFlow model with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving).
* TensorFlow Serving expects models in the [SavedModel](https://www.tensorflow.org/guide/saved_model) format. 
  * SavedModel (Protobuf based) is the default model format for Keras/TensorFlow 2x. 
  * TensorFlow Serving cannot serve models stored in the legacy HD5 format of Keras/TensorFlow 1x.
* MLflow currently can only log a Keras model in the HD5 format. It cannot log a model as a SavedModel flavor.
  * See MLflow git issues [3224](https://github.com/mlflow/mlflow/issues/3224) and [3226](https://github.com/mlflow/mlflow/issues/3226) that address this problem.

In order to serve an MLflow model with TensorFlow Serving you need to convert it to the SavedModel format.
There are two options:
  * Save the model as an artifact in the SavedModel format.
  * Log model as MLflow Keras flavor (HD5) and then convert it to SavedModel format before deploying to TensorFlow Serving.

In this example, we opt for the former as it imposes less of a burden on the downstream MLOps CI/CD deployment pipeline.

Save the model in SavedModel format:
```
  import tensorflow as tf
  tf.keras.models.save_model(model, "tensorflow-model", overwrite=True, include_optimizer=True)
  mlflow.log_artifact("tensorflow-model")
```

You will now use this directory to load the model into TensorFlow Serving.
```
/opt/mlflow/mlruns/1/7e674524514846799310c41f10d6b99d/artifacts/tensorflow-model

 +-variables/
      | +-variables.data-00000-of-00001
      +-saved_model.pb
      +-assets
```


#### Launch TensorFlow Serving as docker container

Set following convenience environment variables and get path to the MLflow model artifact.
```
HOST_PORT=8502
MODEL=keras_mnist
CONTAINER=tfs_serving_$MODEL
DOCKER_MODEL_PATH=/models/$MODEL/01

RUN_ID=7e674524514846799310c41f10d6b99d
HOST_MODEL_PATH=`mlflow artifacts download --run-id $RUN_ID --artifact-path tensorflow-model`
```

##### Generic Image with Mounted Volume

You need to mount the model directory as a container volume.

See TensorFlow documentation: [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).

```
docker run -t --rm --publish $HOST_PORT=8501 \
  --volume $HOST_MODEL_PATH:$DOCKER_MODEL_PATH \
  --env MODEL_NAME=$MODEL \
  tensorflow/serving
```

##### Custom image

With a custom image, you bake the model into the container itself with no external mounted volumes.

See TensorFlow documentation: [Creating your own serving image](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image).

```
BASE_CONTAINER=tfs_serving_base
docker run -d --name $BASE_CONTAINER tensorflow/serving
docker cp $HOST_MODEL_PATH/ $BASE_CONTAINER:/tmp
docker exec -d $BASE_CONTAINER mkdir -p /models/$MODEL
docker exec -d $BASE_CONTAINER mv /tmp/tensorflow-model /models/$MODEL/01
docker commit --change "ENV MODEL_NAME $MODEL" $BASE_CONTAINER $CONTAINER
docker rm -f $BASE_CONTAINER
docker run -d --name $CONTAINER -p $HOST_PORT:8501 $CONTAINER
```

#### Score

Data: [../../data/score/mnist/mnist-tf-serving.json](../../data/score/mnist/mnist-tf-serving.json).

```
curl http://localhost:8502/v1/models/keras_mnist:predict -X POST \
  -d @../../data/score/mnist/mnist-tf-serving.json 
```
```
{
  "predictions": [
    [
      3.12299653e-06,
      2.60791438e-07,
      0.000399814453,
      0.000576042803,
      3.31057066e-08,
      1.12317866e-05,
      1.57459468e-09,
      0.998985946,
      9.80186451e-06,
      1.3647702e-05
    ],
. . .
}
```

## Autologging

There are several autologging options for Keras models:
* mlflow_autolog - calls mlflow.autolog() - globally scoped autologging - same as mlflow.tensorflow.autolog().
* tensorflow_autolog - calls mlflow.tensorflow.autolog().
* keras_autolog - calls mlflow.keras.autolog() - fails for TensorFlow 2.x. Seems redundant.

Interestingly, they behave differently depending on the TensorFlow version.

| TensorFlow Version | Autolog Method | Result | 
|---|---|---|
| 2x | mlflow.mlflow.autolog | OK |
| 2x | mlflow.tensorflow.autolog | OK |
| 2x | mlflow.keras.autolog | ModuleNotFoundError: No module named 'keras' | 
| 1x | mlflow.tensorflow.autolog | none |
| 1x | mlflow.keras.autolog | OK | 


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

MLflow  Autologging

```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --mlflow_autolog True
```

TensorFlow Autologging

```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --tensorflow_autolog True
```

Keras Autologging
```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --keras_autolog True
```
```
ModuleNotFoundError: No module named 'keras'
```

## Testing
```
py.test -s -v test.py
```
```
4 passed, 21 warnings in 71.57s (0:01:11) 
```
