# mlflow-examples - sparkml

## Overview

* PySpark Spark ML Decision Tree Classification example
* Saves model in SparkML and MLeap flavors - optionally also ONNX flavor.
* Demonstrates both batch and real-time scoring.
* Data: [../../data/train/wine-quality-white.csv](../../data/train/wine-quality-white.csv)

## Train

Two model artifacts are created: `spark-model` and `mleap-model`. 
To create an `onnx-model` pass the `--log_as_onnx` option.

### Arguments

|Name | Required | Default | Description|   
|---|---|---|---|
| experiment_name | no | none | Experiment name  |   
| model_name | no | none | Registered model name (if set) |
| data_path | no | ../../data/train/wine-quality-white.csv | Path to data  |
| max_depth | no | 5 | Max depth  |
| max_bins | no | 32 | Max bins  |
| run_origin | no | none | Run tag  |
| log_as_onnx | no | False | Also log the model in ONNX format |

### Unmanaged without mlflow run

To run with standard main function
```
spark-submit --master local[2] \
  --packages com.databricks:spark-avro_2.11:3.0.1,ml.combust.mleap:mleap-spark_2.11:0.12.0 \
  train.py --max_depth 16 --max_bins 32 
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow run` ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

**mlflow run local**
```
mlflow run . \
  -P max_depth=3 -P max_bins=24 \
  --experiment-name=sparkml
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/sparkml \
   -P max_depth=3 -P max_bins=24 \
  --experiment-name=sparkml
```

## Predictions

You can make predictions in two ways:
* Batch predictions 
* Real-time predictions - use MLflow's scoring server to score individual requests.


### Batch Predictions

#### Predict with Spark using SparkML and MLeap flavors

See [spark_predict.py](spark_predict.py).
Predict with Spark using the `spark-model` and `mleap-model` (using MLeap's SparkBundle).

```
spark-submit --master local[2] spark_predict.py \
  --run_id ffd36a96dd204ac38a58a00c94390649
```

```
model_uri: runs:/ffd36a96dd204ac38a58a00c94390649/spark-model
Spark ML predictions
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
. . .
+-----------------+-------+--------------------------------------------------------+

model_uri: runs:/ffd36a96dd204ac38a58a00c94390649/mleap-model
MLeap predictions
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
. . .
+-----------------+-------+--------------------------------------------------------+
```

#### Predict as Pyfunc/Spark flavor

See [pyfunc_predict.py](pyfunc_predict.py).

```
spark-submit --master local[2] pyfunc_predict.py \
  --model_uri runs:/ffd36a96dd204ac38a58a00c94390649/spark-model
```

```
model: <mlflow.spark._PyFuncModelWrapper object at 0x115f30b70>
data.shape: (4898, 12)
predictions: [5.470588235294118, 5.470588235294118, 5.769607843137255, 5.877049180327869, 5.877049180327869]
predictions.len: 4898
```

#### Predict as Pyfunc/ONNX flavor

```
python pyfunc_predict.py \
  --model_uri runs:/ffd36a96dd204ac38a58a00c94390649/onnx-model
```
Fails. Apparently the ONNX pyfunc code doesn't support columns with spaces.
```
KeyError: 'fixed_acidity'
```
If we change the spaces to underscores, we get another error.
```
ONNXRuntimeError: INVALID_ARGUMENT : Invalid rank for input: sulphates Got: 1 Expected: 2 Please fix either the inputs or the model.
```

#### Predict as ONNX flavor

Scores directly with ONNX runtime - no Spark needed.
See [onnx_predict.py](onnx_predict.py). 

```
python onnx_predict.py \
  --model_uri runs:/ffd36a96dd204ac38a58a00c94390649/spark-model
```
```
ONNXRuntimeError: INVALID_ARGUMENT : Invalid rank for input: sulphates Got: 1 Expected: 2 Please fix either the inputs or the model.
```

### Real-time Predictions

Use a server to score predictions over HTTP.

There are several ways to launch the server:
  1. MLflow scoring web server 
  2. Plain docker container
  3. SageMaker docker container
  4. Azure docker container

See MLflow documentation:
* [Built-In Deployment Tools](https://mlflow.org/docs/latest/models.html#built-in-deployment-tools)
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)

In one window launch the server.

In another window, score some data.
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/score/wine-quality.json \
  http://localhost:5001/invocations
```
```
[
  [5.470588235294118, 5.470588235294118, 5.769607843137255]
]
```

Data should be in `JSON-serialized Pandas DataFrames split orientation` format
such as [score/wine-quality.json](../../data/predict-wine-quality.json).
```
{
  "columns": [
    "alcohol",
    "chlorides",
    "citric acid",
    "density",
    "fixed acidity",
    "free sulfur dioxide",
    "pH",
    "residual sugar",
    "sulphates",
    "total sulfur dioxide",
    "volatile acidity"
  ],
  "data": [
    [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8, 6 ],
    [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5, 6 ],
    [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1, 6 ]
  ]
}
```

#### 1. MLflow scoring web server

Launch the web server.
```
mlflow pyfunc serve -port 5001 \
  -model-uri runs:/7e674524514846799310c41f10d6b99d/spark-model 
```

Make predictions with curl as described above.

#### 2. Plain Docker Container

See [build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) documentation.

First build the docker image.
```
mlflow models build-docker \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/spark-model \
  --name dk-wine-sparkml
```

Then launch the server as a docker container.
```
docker run --p 5001:8080 dk-wine-sparkml
```
Make predictions with curl as described above.

#### 3. SageMaker Docker Container

See documentation:
* [Deploy a python_function model on Amazon SageMaker](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-amazon-sagemaker)
* [mlflow.sagemaker](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html)

Notes:
  * You can test your SageMaker container on your local machine before pushing to SageMaker.
  * You can build a container either with the Spark or MLeap model. 


##### Score with Spark ML model

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-sparkml-spark
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/spark-model \
  --port 5001 --image sm-wine-sparkml-spark
```
##### Score with MLeap model

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-sparkml-mleap
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/mleap-model \
  --port 5001 --image sm-wine-sparkml-mleap
```

Make predictions with curl as described above.

#### 4. Azure docker container

See [Deploy a python_function model on Microsoft Azure ML](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-microsoft-azure-ml) documentation.

TODO.

