# mlflow-examples - sparkml

## Overview

* PySpark Spark ML Decision Tree Classification example
* Logs model in SparkML, custom UDF, MLeap and ONNX flavor.
* Demonstrates both batch (UDF) and real-time scoring.
* Data: [../../data/train/wine-quality-white.csv](../../data/train/wine-quality-white.csv)

## Train

The model can be logged in the following flavors:
* spark-model - Always logged
* mleap-model - Use the `log_as_mleap` option
* onnx-model - Use the `log_as_onnx` option

### Arguments

|Name | Required | Default | Description|   
|---|---|---|---|
| experiment_name | no | none | Experiment name  |   
| model_name | no | none | Registered model name (if set) |
| data_path | no | ../../data/train/wine-quality-white.csv | Path to data  |
| max_depth | no | 5 | Max depth  |
| max_bins | no | 32 | Max bins  |
| run_origin | no | none | Run tag  |
| log_as_mleap | no | False | Log the model in MLeap flavor |
| log_as_onnx | no | False | Log the model in ONNX flavor |
| spark_autolog | no | False | [Spark autologging](https://www.mlflow.org/docs/latest/tracking.html#spark-experimental) with Spark 3.x |

### Run unmanaged without `mlflow run`

Install [conda.yaml](conda.yaml) environment.

To run with standard main function.
```
spark-submit --master local[2] \
  train.py --max_depth 16 --max_bins 32 
```

To log model as MLeap.
```
spark-submit --master local[2] \
  --packages com.databricks:spark-avro_2.11:3.0.1,ml.combust.mleap:mleap-spark_2.11:0.12.0 \
  train.py --log_as_mleap True
```

To log model as ONNX.
```
spark-submit --master local[2] \
  train.py --log_as_onnx True
```

Spark autologging works only with Spark 3.x and Scala 2.12.
It logs the data source in the tag `sparkDatasourceInfo`.
```
spark-submit --master local[2] \
  --packages org.mlflow:mlflow-spark:1.8.0 \
  train.py --spark_autolog True
```

Resulting tag:
* Tag name: `sparkDatasourceInfo`
* Value: `path=file:/var/folders/rp/88lfxw2n4lvgkdk9xl0lkqjr0000gp/T/DecisionTreeRegressionModel_1590801461.6367931/data,format=parquet`

### Using `mlflow run`

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
mlflow run https://github.com/amesar/mlflow-examples.git#python/sparkml \
   -P max_depth=3 -P max_bins=24 \
  --experiment-name=sparkml
```

## Predictions

You can make predictions in two ways:
* Batch predictions 
* Real-time predictions - use MLflow's scoring server


### Batch Predictions

Flavors:
* Spark ML
* UDF
* MLeap
* ONNX
* PyFunc
  * PyFunc - SparkML
  * PyFunc - ONNX

#### Predict as SparkML flavor

Predict as Spark ML flavor with `spark-model`.
See [spark_predict.py](spark_predict.py).

```
spark-submit --master local[2] spark_predict.py \
  --model_uri runs:/ffd36a96dd204ac38a58a00c94390649/spark-model
```

```
model.type: <class 'pyspark.ml.pipeline.PipelineModel'>
predictions.type: <class 'pyspark.sql.dataframe.DataFrame'>
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
. . .
+-----------------+-------+--------------------------------------------------------+
```

#### Predict as UDF

Predict as Spark ML UDF with `udf-spark-model`.
See [udf_predict.py](udf_predict.py).

There is a bug with loading a Spark ML model as a UDF. If you try to load `spark-model` you will get:
```
java.lang.IllegalArgumentException: Field "fixed acidity" does not exist.
Available fields: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```

You need to log the Spark ML model as `udf-spark-model` which uses a custom PythonModel that handles the column numeric problem.
See [pyspark_udf_workaround.py](pyspark_udf_workaround.py).

```
spark-submit --master local[2] udf_predict.py \
  --model_uri runs:/ffd36a96dd204ac38a58a00c94390649/udf-spark-model
```

```
predictions.type: <class 'pyspark.sql.dataframe.DataFrame'>
+-------+------------------+
|quality|prediction        |
+-------+------------------+
|6      |5.4586894586894585|
|6      |5.011627906976744 |
+-------+------------------+
```

Note: [pyarrow version](conda.yaml) must be 0.13.0 (or lower) for UDF prediction. Otherwise you get the following misleading error.
```
ImportError: PyArrow >= 0.8.0 must be installed; however, it was not found.
```

#### Predict as MLeap flavor

Predict as MLeap flavor using MLeap's SparkBundle with `mleap-model`.
See [mleap_predict.py](mleap_predict.py).
```
spark-submit --master local[2] mleap_predict.py \
  --run_id ffd36a96dd204ac38a58a00c94390649
```

```
model_uri: runs:/ffd36a96dd204ac38a58a00c94390649/mleap-model
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
model.type: <mlflow.spark._PyFuncModelWrapper object at 0x115f30b70>
predictions.type: <class 'list'>
predictions.len: 4898
predictions: [5.470588235294118, 5.470588235294118, 5.769607843137255, 5.877049180327869, 5.877049180327869]
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
