# mlflow-examples - sklearn 

## Overview
* Wine Quality Decision Tree Example
* Is a well-formed Python project that generates a wheel
* This example demonstrates all features of MLflow training and prediction.
* Saves model in pickle format
* Saves plot artifacts
* Shows several ways to run training:
  * _mlflow run_ - several variants
  * Run against Databricks cluster 
  * Call wheel from notebook, etc.
* Shows several ways to run prediction  
  * web server
  * mlflow.load_model()
  *  UDF - invoke with the DataFrame API or SQL
* Data: [../../data/train/wine-quality-white.csv](../../data/train/wine-quality-white.csv)

## Training

Source: [main.py](main.py) and [wine_quality/train.py](wine_quality/train.py).

There are several ways to train a model with MLflow.
  1. Unmanaged without MLflow CLI
  1. MLflow CLI `run` command
  1. Databricks REST API

### Arguments

|Name | Required | Default | Description| 
|---|---|---|---|
| experiment_name | no | none | Experiment name  |  
| model_name | no | none | Registered model name (if set) |  
| data_path | no | ../../data/train/wine-quality-white.csv | Path to data  |  
| max_depth | no | none | Max depth  |  
| max_leaf_nodes | no | none | Max leaf nodes  |  
| run_origin | no | none | Run tag  |  
| log_as_onnx | no | False | Also log the model in ONNX format |  

### 1. Unmanaged without MLflow CLI

Run the standard main function from the command-line.
```
python main.py --experiment_name sklearn --max_depth 2 --max_leaf_nodes 32
```

### 2. MLflow CLI - `mlflow run`

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that `mlflow` CLI run ignores the `set_experiment()` so you must specify the experiment with the  `--experiment-sklearn` argument.

#### mlflow run local
```
mlflow run . \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=localRun \
  --experiment-name=sklearn_wine
```

#### mlflow run github
```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=gitRun \
  --experiment-name=sklearn_wine
```

#### mlflow run Databricks remote

Run against a Databricks cluster.
You will need a cluster spec file such as [mlflow_run_cluster.json](mlflow_run_cluster.json).
See MLflow [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-databricks) page 

Setup - set MLFLOW_TRACKING_URI.
```
export MLFLOW_TRACKING_URI=databricks
```

Setup - build the wheel and push it to the Databricks file system.
```
python setup.py bdist_wheel
databricks fs cp \
  dist/mlflow_sklearn_wine-0.0.1-py3-none-any.whl \
  dbfs:/tmp/jobs/sklearn_wine/mlflow_wine_quality-0.0.1-py3.6.whl 
databricks fs cp data/train/wine-quality-white.csv dbfs:/tmp/jobs/sklearn_wine/wine-quality-white.csv
```
The token and tracking server URL will be picked up from your Databricks CLI ~/.databrickscfg default profile.

Now run the model.
```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=gitRun \
  -P data_path=/dbfs/tmp/data/wine-quality-white.csv \
  --experiment-name=/Users/juan.doe@acme.com/sklearn_wine \
  --backend databricks --backend-config mlflow_run_cluster.json
```

### 3. Databricks REST API

You can also package your code as a wheel and run it with the standard Databricks REST API endpoints
[job/runs/submit](https://docs.databricks.com/api/latest/jobs.html#runs-submit) 
or [jobs/run-now](https://docs.databricks.com/api/latest/jobs.html#run-now) 
using the [spark_python_task](https://docs.databricks.com/api/latest/jobs.html#jobssparkpythontask). 

#### Setup

First build the wheel.
```
python setup.py bdist_wheel
```

Upload the data file, main file and wheel to your Databricks file system.
```
databricks fs cp main.py dbfs:/tmp/jobs/sklearn_wine/main.py
databricks fs cp data/train/wine-quality-white.csv dbfs:/tmp/jobs/sklearn_wine/wine-quality-white.csv
databricks fs cp \
  dist/mlflow_sklearn_wine-0.0.1-py3-none-any.whl \
  dbfs:/tmp/jobs/sklearn_wine/mlflow_wine_quality-0.0.1-py3.6.whl 
```


#### Run Submit

##### Run with new cluster

Define your run in [run_submit_new_cluster.json](run_submit_new_cluster.json) and launch the run.

```
databricks runs submit --json-file run_submit_new_cluster.json
```

##### Run with existing cluster

Every time you build a new wheel, you need to upload (as described above) it to DBFS and restart the cluster.
```
databricks clusters restart --cluster-id 1222-015510-grams64
```

Define your run in [run_submit_existing_cluster.json](run_submit_existing_cluster.json) and launch the run.
```
databricks runs submit --json-file run_submit_existing_cluster.json
```

#### Job Run Now

##### Run with new cluster

First create a job with the spec file [create_job_new_cluster.json](create_job_new_cluster.json). 
```
databricks jobs create --json-file create_job_new_cluster.json
```

Then run the job with desired parameters.
```
databricks jobs run-now --job-id $JOB_ID \
  --python-params '[ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/sklearn_wine/wine-quality-white.csv" ]'
```

##### Run with existing cluster
First create a job with the spec file [create_job_existing_cluster.json](create_job_existing_cluster.json).
```
databricks jobs create --json-file create_job_existing_cluster.json
```

Then run the job with desired parameters.
```
databricks jobs run-now --job-id $JOB_ID --python-params ' [ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/sklearn_wine/wine-quality-white.csv" ] '
```

#### Run wheel from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
from wine_quality import Trainer
data_path = "/dbfs/tmp/jobs/sklearn_wine/wine-quality-white.csv"
trainer = Trainer("WineQualityExperiment", data_path, "from_notebook_with_wheel")
trainer.train(0.4, 0.4)
```

## Predictions

You can make predictions in two ways:
* Batch predictions - direct calls to retrieve the model and score large files.
  * mlflow.sklearn.load_model()
  * mlflow.pyfunc.load_model()
  * Spark UDF - either the DataFrame API or SQL
* Real-time predictions - use MLflow's scoring server to score individual requests.


### Batch Predictions

#### 1. Predict with mlflow.sklearn.load_model()

You can use either a `runs` or `models` scheme.

URI with `runs` scheme.
```
python sklearn_predict.py runs:/7e674524514846799310c41f10d6b99d/sklearn-model
```

URI with `models` scheme.
Assume you have a registered model with a production stage or version 1.
```
python sklearn_predict.py models:/sklearn_wine/production
python sklearn_predict.py models:/sklearn_wine/1
```

Result.
```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```


Snippet from [sklearn_predict.py](sklearn_predict.py):
```
model = mlflow.sklearn.load_model(model_uri)
df = pd.read_csv("../../data/train/wine-quality-white.csv")
predictions = model.predict(data)
```


#### 2. Predict with mlflow.pyfunc.load_model()

```
python pyfunc_predict.py runs:/7e674524514846799310c41f10d6b99d/sklearn-model
```

```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](pyfunc_predict.py):
```
data_path = "../../data/train/wine-quality-white.csv"
data = util.read_prediction_data(data_path)
model_uri = client.get_run(run_id).info.artifact_uri + "/sklearn-model"
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)

```

#### 3. Predict with Spark UDF (user-defined function)

See [Export a python_function model as an Apache Spark UDF]((https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf) documentation.

We show how to invoke a UDF with both the DataFrame API and SQL.

Scroll right to see prediction column.

```
pip install pyarrow

spark-submit --master local[2] spark_udf_predict.py \
  runs:/7e674524514846799310c41f10d6b99d/sklearn-model
```

```
+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|alcohol|chlorides|citric acid|density|fixed acidity|free sulfur dioxide|  pH|residual sugar|sulphates|total sulfur dioxide|volatile acidity|        prediction|
+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|    8.8|    0.045|       0.36|  1.001|          7.0|               45.0| 3.0|          20.7|     0.45|               170.0|            0.27| 5.551096337521979|
|    9.5|    0.049|       0.34|  0.994|          6.3|               14.0| 3.3|           1.6|     0.49|               132.0|             0.3| 5.297727513113797|
4   10.1|     0.05|        0.4| 0.9951|          8.1|               30.0|3.26|           6.9|     0.44|                97.0|            0.28| 5.427572126267637|
|    9.9|    0.058|       0.32| 0.9956|          7.2|               47.0|3.19|           8.5|      0.4|               186.0|            0.23| 5.562886443251915|
```
From [spark_udf_predict.py](spark_udf_predict.py):
```
spark = SparkSession.builder.appName("App").getOrCreate()
data = spark.read.option("inferSchema",True).option("header", True).csv("../data/train/wine-quality-white.csv")
data = data.drop("quality")
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = data.withColumn("prediction", udf(*df.columns))
predictions.show(10)

spark.udf.register("predictUDF", udf)
data.createOrReplaceGlobalTempView("data")
predictions = spark.sql("select *, predictUDF(*) as prediction from global_temp.data")
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
  [5.470588235294118,5.470588235294118,5.769607843137255]
]
```

Data should be in `JSON-serialized Pandas DataFrames split orientation` format
such as [score/wine-quality.json](../../data/score/wine-quality.json).
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
  -model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model 
```

Make predictions with curl as described above.

#### 2. Plain Docker Container

See [build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) documentation.

First build the docker image.
```
mlflow models build-docker \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model \
  --name dk-wine-sklearn
```

Then launch the server as a docker container.
```
docker run --p 5001:8080 dk-wine-sklearn
```
Make predictions with curl as described above.

#### 3. SageMaker Docker Container

See documentation:
* [Deploy a python_function model on Amazon SageMaker](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-amazon-sagemaker)
* [mlflow.sagemaker](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html)

You can test your SageMaker container on your local machine before pushing to SageMaker.

First build the docker image.
```
mlflow sagemaker build-and-push-container --build --no-push --container sm-wine-sklearn
```

To test locally, launch the server as a docker container.
```
mlflow sagemaker run-local \
  --model-uri runs:/7e674524514846799310c41f10d6b99d/sklearn-model \
  --port 5001 --image sm-wine-sklearn
```

Make predictions with curl as described above.

#### 4. Azure docker container

See [Deploy a python_function model on Microsoft Azure ML](https://mlflow.org/docs/latest/models.html#deploy-a-python-function-model-on-microsoft-azure-ml) documentation.

TODO.

