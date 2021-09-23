# mlflow-examples - sparkml

## Overview

* PySpark Spark ML Decision Tree Classification example.
* Logs model in SparkML, custom UDF, MLeap and ONNX flavors.
* Both batch and real-time scoring.
* Data: [../../data/train/wine-quality-white.csv](../../data/train/wine-quality-white.csv)

## Train

The model can be logged in the following flavors:
* spark-model - Always logged.
* udf-spark-model - Custom Pyfunc model for spark-model UDF option.
* onnx-model - Use the `log-as-onnx` option.
* mleap-model - Use the `log-as-mleap` option.

### Options

```
python train.py --help

Options:
  --experiment-name TEXT   Experiment name
  --data-path TEXT         Data path
  --model-name TEXT        Registered model name
  --max-depth INTEGER      Max depth
  --max-bins INTEGER       Max bins
  --describe BOOLEAN       Describe data
  --log-as-mleap BOOLEAN   Score as MLeap
  --log-as-onnx BOOLEAN    Log model as ONNX flavor
  --spark-autolog BOOLEAN  Use spark.autolog
```

### Run with `mlflow run`

See the [MLproject](MLproject) file for the main entrypoints. 
For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow run` ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-id` argument.

To run with the default main function.
```
mlflow run . -P max-depth=16 -P max-bins=32 \
  -P model-name=sparkml \
  --experiment-name sparkml
```
To log model as ONNX.
```
mlflow run . -P log-as-onnx=True \
  -P model-name=sparkml \
  --experiment-name sparkml
```

**mlflow run github with Databricks**

```
export MLFLOW_TRACKING_URI=databricks
```
```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sparkml \
  -P max-depth=3 -P max-bins=24 \
  -P data-path=https://raw.githubusercontent.com/amesar/mlflow-examples/master/data/train/wine-quality-white.csv \
  --experiment-name=/Users/me@mycompany.com/experiments/sparkml
  --backend databricks --backend-config mlflow_run_cluster.json
```

### Run without `mlflow run`

Install [conda.yaml](conda.yaml) environment.

#### Log model as MLeap.
```
spark-submit --master local[2] \
  --packages com.databricks:spark-avro_2.11:3.0.1,ml.combust.mleap:mleap-spark_2.11:0.12.0 \
  train.py --log-as-mleap True
```

#### Log model with Autologging.

Spark autologging works only with Spark 3.x and Scala 2.12.
It logs the data source in the tag `sparkDatasourceInfo`.
```
spark-submit --master local[2] \
  --packages org.mlflow:mlflow-spark:1.12.1 \
  train.py --spark-autolog True
```

Resulting tag:
* Tag name: `sparkDatasourceInfo`
* Value: `path=file:/var/folders/rp/88lfxw2n4lvgkdk9xl0lkqjr0000gp/T/DecisionTreeRegressionModel_1590801461.6367931/data,format=parquet`


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

Predict as native Spark ML flavor.
See [spark_predict.py](spark_predict.py).

```
mlflow run . --entry-point spark_predict \
  -P model-uri=models:/sparkml/1
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
#### Predict as Pyfunc/SparkML flavor

See [pyfunc_predict.py](pyfunc_predict.py).

```
mlflow run . --entry-point pyfunc_predict \
  -P model-uri=models:/sparkml/1
```

```
model.type: <mlflow.spark._PyFuncModelWrapper object at 0x115f30b70>
predictions.type: <class 'list'>
predictions.len: 4898
predictions: [5.470588235294118, 5.470588235294118, 5.769607843137255, 5.877049180327869, 5.877049180327869]
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
See [sparkml_udf_workaround.py](sparkml_udf_workaround.py).

```
mlflow run . --entry-point udf_predict \
  -P model-uri=models:/sparkml/1
```

```
predictions.type: <class 'pyspark.sql.dataframe.DataFrame'>
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.497191011235955|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|4.833333333333333|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
|5.670157068062827|6      |[8.1,0.28,0.4,6.9,0.05,30.0,97.0,0.9951,3.26,0.44,10.1] |
+-----------------+-------+--------------------------------------------------------+
```

#### Predict as MLeap flavor

Predict as MLeap flavor using MLeap's SparkBundle with `mleap-model`.
See [mleap_predict.py](mleap_predict.py).
```
spark-submit --master local[2] mleap_predict.py \
  --run_id ffd36a96dd204ac38a58a00c94390649/mleap-model
```

```
+-----------------+-------+--------------------------------------------------------+
|prediction       |quality|features                                                |
+-----------------+-------+--------------------------------------------------------+
|5.470588235294118|6      |[7.0,0.27,0.36,20.7,0.045,45.0,170.0,1.001,3.0,0.45,8.8]|
|5.470588235294118|6      |[6.3,0.3,0.34,1.6,0.049,14.0,132.0,0.994,3.3,0.49,9.5]  |
. . .
+-----------------+-------+--------------------------------------------------------+
```

#### Predict as Pyfunc/ONNX flavor

```
python pyfunc_predict.py --model-uri models/sparkml_onnx/1
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
  --model-uri models:/sparkml_onnx/1
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
  -model-uri models:/sparkml/1
```

Make predictions with curl as described above.

#### 2. Plain Docker Container

See [build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) documentation.

First build the docker image.
```
mlflow models build-docker \
  --model-uri models:/sparkml/1
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
  --model-uri models:/sparkml/1 \
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
  --model-uri models:/sparkml/1 \
  --port 5001 --image sm-wine-sparkml-mleap
```

Make predictions with curl as described above.

#### 4. Azure docker container

See [Deploy a python_function model on Microsoft Azure ML](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-microsoft-azure-ml) documentation.
