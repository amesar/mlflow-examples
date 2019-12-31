# mlflow-examples - pyspark

## Overview

* PySpark Spark ML Decision Tree Classification example
* Saves model in SparkML and MLeap flavors - optionally also ONNX flavor.
* Demonstrates both batch and real-time scoring.
* Data: [../data/wine-quality-white.csv](../data/wine-quality-white.csv)

## Train

Two model artifacts are created: `spark-model` and `mleap-model`. 
To create an `onnx-model` pass the `--log_as_onnx` option.

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
  --experiment-name=pyspark
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/pyspark \
   -P max_depth=3 -P max_bins=24 \
  --experiment-name=pyspark
```

## Predictions

You can make predictions in two ways:
* Batch predictions 
* Real-time predictions - use MLflow's scoring server to score individual requests.


### Batch Predictions

#### Predict as Spark and MLeap flavors

See [spark_predict.py](spark_predict.py).

```
spark-submit --master local[2] spark_predict.py \
  _id7b951173284249f7a3b27746450ac7b0
```

```
Spark ML predictions
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
|5.769607843137255|6      |[8.1,0.28,0.4,6.9,0.05,30.0,97.0,0.9951,3.26,0.44,10.1] |
|5.877049180327869|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
|5.877049180327869|6      |[7.2,0.23,0.32,8.5,0.058,47.0,186.0,0.9956,3.19,0.4,9.9]|
+-----------------+-------+--------------------------------------------------------+
only showing top 5 rows

model_uri: runs:/ffd36a96dd204ac38a58a00c94390649/mleap-model
MLeap ML predictions
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
. . .
```

#### Predict as Pyfunc/Spark flavor

See [pyfunc_predict.py](pyfunc_predict.py).

```
spark-submit --master local[2] pyfunc_predict.py \
  runs:/7b951173284249f7a3b27746450ac7b0/spark-model
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
  runs:/7b951173284249f7a3b27746450ac7b0/onnx-model
```
Fails. Apparently the ONNX pyfunc code doesn't support columns with spaces.
```
KeyError: 'fixed_acidity'
```
If we change the spaces to underscores, we get another error.
```
RuntimeError: Method run failed due to: [ONNXRuntimeError] : 1 : GENERAL ERROR : /Users/vsts/agent/2.148.0/work/1/s/onnxruntime/core/providers/common.h:18 int64_t onnxruntime::HandleNegativeAxis(int64_t, int64_t) axis >= -tensor_rank && axis <= tensor_rank - 1 was false. axis 1 is not in valid range [-1,0]
```

#### Predict as ONNX flavor

Scores directly with ONNX runtime - no Spark needed.
See [onnx_predict.py](onnx_predict.py). 

```
python onnx_predict.py \
  --model_uri runs:/7b951173284249f7a3b27746450ac7b0/spark-model
```
```
model.type: <class 'onnx.onnx_ONNX_REL_1_4_ml_pb2.ModelProto'>
predictions.type: <class 'numpy.ndarray'>
predictions.shape: (4898, 1)
predictions: [
 [5.470588 ]
 [5.470588 ]
 ...
 [6.75     ]
 [6.25     ]]
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
  -d @../data/predict-wine-quality.json \
  http://localhost:5001/invocations
```
```
[
  [5.470588235294118, 5.470588235294118, 5.769607843137255]
]
```

Data should be in `JSON-serialized Pandas DataFrames split orientation` format
such as [predict-wine-quality.json](../data/predict-wine-quality.json).
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
  --name dk-wine-pyspark
```

Then launch the server as a docker container.
```
docker run --p 5001:8080 dk-wine-pyspark
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
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-pyspark-spark
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/spark-model \
  --port 5001 --image sm-wine-pyspark-spark
```
##### Score with MLeap model

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-pyspark-mleap
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/mleap-model \
  --port 5001 --image sm-wine-pyspark-mleap
```

Make predictions with curl as described above.

#### 4. Azure docker container

See [Deploy a python_function model on Microsoft Azure ML](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-microsoft-azure-ml) documentation.

TODO.

