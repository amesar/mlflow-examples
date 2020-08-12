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

### Autologging

To run with user logging (no autologging).
```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128
```
or
```
mlflow run . --experiment-name keras_mnist -P epochs=3 -P batch_size=128
```

### ONNX

To log a model as ONNX flavor under the artifact path `onnx-model`.
```
mlflow run . --experiment-name keras_mnist -P epochs=3 -P batch_size=128 -P log_as_onnx=True
```

## Batch Scoring

### Score as Keras and PyFunc flavor 

Score as Keras and Keras/PyFunc flavor.
Source: [keras_predict.py](keras_predict.py).

```
python keras_predict.py --model_uri runs:/7e674524514846799310c41f10d6b99d/keras-model
```

```
**** mlflow.keras.load_model
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (10000,)
predictions: [7 2 1 ... 4 5 6]

**** mlflow.pyfunc.load_model
model.type: <class 'mlflow.keras._KerasModelWrapper'>
predictions.type: <class 'pandas.core.frame.DataFrame'>
predictions.shape: (10000, 10)
predictions:                  0             1  ...             8             9
0     1.263480e-06  6.968530e-08  ...  2.910662e-06  3.157784e-05
. . .
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

## Real-time Scoring

Two real-time scoring server options are demonstrated here:
* MLflow scoring server
* TensorFlow Servering scoring server

### Real-time Scoring Data

Corresponding request data:
* MLflow scoring server - [../../data/score/mnist/mnist-mlflow.json](../../data/score/mnist/mnist-mlflow.json)
* TensorFlow Serving - [../../data/score/mnist/mnist-tf-serving.json](../../data/score/mnist/mnist-tf-serving.json).


The above request data is generated from a JSON dump of reshaped MNIST data. 
For details see [create_scoring_datafiles.py](create_scoring_datafiles.py).

### Real-time Scoring - MLflow

You can launch the the MLflow scoring server either as:
* Local web server 
* Docker container

#### Data

[../../data/score/mnist/mnist-mlflow.json](../../data/score/mnist/mnist-mlflow.json)

#### 1. Local Web server

```
mlflow pyfunc serve -port 5001 \
  -model-uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model
```

#### 2. Docker container

You can use the run SageMaker container on your local machine.

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-mnist-keras
```

Then launch the server in a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/keras-hd5-model \
  --port 5001 --image sm-mnist-keras
```

#### Score
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


### TensorFlow Serving Real-time Scoring

Here we demonstrate how to serve a TensorFlow/Keras MLflow  model with TensorFlow Serving.

#### Launch TensorFlow scoring server as docker container

##### Generic

See [TensorFlow Serving with Docker](https://www.tensorflow.org/tfx/serving/docker).

```
docker run -t --rm --publish 8502 \
--volume /opt/mlflow/mlruns/1/f48dafd70be044298f71488b0ae10df4/artifacts/tensorflow-model:/models/keras_mnist\
--env MODEL_NAME=keras_mnist \
tensorflow/serving
```

##### Custom image

See [Creating your own serving image](https://www.tensorflow.org/tfx/serving/docker#creating_your_own_serving_image).

```
HOST_PORT=8502
MODEL=keras_mnist
CONTAINER=tfs_serving_custom_$MODEL
BASE_CONTAINER=tfs_serving_base

HOST_MODEL_PATH=$PWD/tensorflow_serving_models/keras_mnist
DOCKER_MODEL_PATH=/models/$MODEL

docker run -d --name $BASE_CONTAINER tensorflow/serving
docker cp $HOST_MODEL_PATH $BASE_CONTAINER:/models/$MODEL
docker commit --change "ENV MODEL_NAME $MODEL" $BASE_CONTAINER $CONTAINER
docker rm -f $BASE_CONTAINER
docker run -d --name $CONTAINER -p $HOST_PORT:8501 $CONTAINER
```

#### Score

See [../../data/score/mnist/mnist-tf-serving.json](../../data/score/mnist/mnist-tf-serving.json).

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

There are two autologging options:
* keras_autolog - calls mlflow.keras.autolog()
* tensorflow_autolog - calls mlflow.tensorflow.autolog()

Interestingly, they behave differently depending on the TensorFlow version.

| TensorFlow Version | Autolog Method | Params | 
|---|---|---|
| 1x | mlflow.keras.autolog | OK | 
| 1x | mlflow.tensorflow.autolog | none |
| 2x | mlflow.keras.autolog | ModuleNotFoundError: No module named 'keras' | 
| 2x | mlflow.tensorflow.autolog | OK |


```
python train.py --experiment_name keras_mnist --epochs 3 --batch_size 128 --keras_autolog True
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
