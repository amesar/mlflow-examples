# mlflow-examples - sklearn 

## Overview
* This example demonstrates all different ways to train and score a MLflow project.
* Uses a Wine Quality DecisionTreeRegressor example.
* Is a well-formed Python project that generates a wheel.
* Saves model in pickle format.
* Saves plot artifacts.
* There are two ways to log your model training:
  * [Manual logging](#Training---Manual-Logging)
    * Shows several ways to run training:
        * `mlflow run` - several variants.
        * Run against a Databricks cluster using `mlflow` run or Databricks job.
        * Run a Databricks job with a wheel built from this package.
  * [Autologging](#Training---Autologging)
* Shows several ways to run predictions:
  * Real-time scoring
    * Local web server.
    * Docker container - Plain docker and AWS SageMaker in local mode.
  * Batch scoring
    * mlflow.load_model()
    * UDF - invoke with the DataFrame API or SQL. Works with Spark 3.1.2.
* Data: [../../data/train/wine-quality-white.csv](../../data/train/wine-quality-white.csv)

## Setup

```
python -m venv mlflow-examples-sklearn
source mlflow-examples-sklearn/bin/activate
pip install -e .
```

## Training - Manual Logging

Source: [wine_quality/train.py](wine_quality/train.py).

There are several ways to train a model with MLflow.
  1. MLflow CLI `run` command
  1. Unmanaged without MLflow CLI
  1. Databricks REST API

### Options

```
python -m wine_quality.train --help

Options:
  --experiment-name TEXT          Experiment name.
  --run-name TEXT                 Run name
  --data-path TEXT                Data path.  [default: https://raw.githubuser
                                  content.com/mlflow/mlflow/master/examples/sk
                                  learn_elasticnet_wine/wine-quality.csv]
  --model-name TEXT               Registered model name.
  --model-version-stage TEXT      Registered model version stage:
                                  production|staging|archive|none.
  --archive-existing-versions BOOLEAN
                                  Archive existing versions.  [default: False]
  --model-alias TEXT              Registered model alias
  --save-signature BOOLEAN        Save model signature. Default is False.
                                  [default: False]
  --log-as-onnx BOOLEAN           Log model as ONNX flavor. Default is false.
                                  [default: False]
  --max-depth INTEGER             Max depth parameter.
  --max-leaf-nodes INTEGER        Max leaf nodes parameter.  [default: 32]
  --run-origin TEXT               Run origin.  [default: none]
  --output-path TEXT              Output file containing run ID.
  --use-run-id-as-run-name BOOLEAN
                                  use_run_id_as_run_name  [default: False]
  --log-evaluation-metrics BOOLEAN
                                  Log metrics from mlflow.evaluate  [default:
                                  False]
  --log-shap BOOLEAN              Log mlflow.shap.log_explanation  [default:
                                  False]
```

#### Signature

See [Model Signature](https://www.mlflow.org/docs/latest/models.html#model-signature) documentation.

If you save the schema of the expected input and output data with a model, you will (usualy) get a better error message in case of a schema mismatch when scoring with pyfunc or real-time scoring server.

Examples:

[ Mixed type](../../data/score/wine-quality/signature_test/json/wine-quality-white-type.json) for column `alcohol`.

| With schema | Error message |
|----------|---------|
| no | ValueError: could not convert string to float: ' 8.8_FOO' |
| yes |  mlflow.exceptions.MlflowException: Incompatible input types for column alcohol. Can not safely convert object to float64. |

[Less columns](../../data/score/wine-quality/ignature_test/json/wine-quality-white-less-columns.json)

| With schema | Error message |
|----------|---------|
| no | ValueError: Number of features of the model must match the input. Model n_features is 11 and input n_features is 10 |
| yes | mlflow.exceptions.MlflowException: Model input is missing columns ['alcohol']. Note that there were extra columns: []
 |

[More columns](../../data/score/wine-quality/signature_test/json/wine-quality-white-more-columns.json)

| With schema | Error message |
|----------|---------|
| no | ValueError: Number of features of the model must match the input. Model n_features is 11 and input n_features is 12 |
| yes | No error |


### 1. Unmanaged without MLflow CLI

Run the standard main function from the command-line.
```
python main.py --experiment-name sklearn --max-depth 2 --max-leaf-nodes 32
```

### 2. MLflow CLI - `mlflow run`

Use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that the `mlflow` CLI run ignores the `mlflow.set_experiment()` so you must specify the experiment with the  `--experiment-name` argument.

#### mlflow run local
```
mlflow run . \
  -P max-depth=2 -P max-leaf-nodes=32 -P run-origin=localRun \
  --experiment-name=sklearn_wine
```

#### mlflow run github
```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  -P max-depth=2 -P max-leaf-nodes=32 -P run-origin=gitRun \
  --experiment-name=sklearn_wine
```

#### mlflow run Databricks remote

Run against a Databricks cluster.
You will need a cluster spec file such as [mlflow_run_cluster.json](mlflow_run_cluster.json).
See MLflow [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-databricks).

Setup:
```
export MLFLOW_TRACKING_URI=databricks
```

The token and tracking server URL are picked up from your Databricks CLI `~/.databrickscfg` default profile.

```
mlflow run https://github.com/amesar/mlflow-examples.git#python/sklearn \
  -P max-depth=2 -P max-leaf-nodes=32 -P run-origin=gitRun \
  -P data-path=https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv \
  --experiment-name=/Users/me@mycompany.com/sklearn_wine \
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
databricks fs cp wine_quality/main_train.py dbfs:/tmp/jobs/sklearn_wine/main.py
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

## Training - Autologging

Source: [wine_quality/autolog_train.py](wine_quality/autolog_train.py).

### Autolog parameters

To activate autologging, you simply call [mlflow.sklearn.autolog()](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.autolog).
Note that the model artifact path is simply `model`.

List of parameters for DecisionTreeRegressor:
* criterion       
* max_depth       
* max_features    
* max_leaf_nodes  
* min_impurity_decrease   
* min_impurity_split      
* min_samples_leaf        
* min_samples_split       
* min_weight_fraction_leaf        
* presort 
* random_state    
* splitter

List of metrics for DecisionTreeRegressor:
* training_mae
* training_mse
* training_r2_score
* training_rmse
* training_score

### Run
```
mlflow run . \
  --entry-point autolog \
  --experiment-name sklearn_autolog\
  -P max-depth=5 
```

### Options
```
Usage: python -m wine_quality.autolog_train [OPTIONS]

Options:
  --experiment-name TEXT    Experiment name.
  --data-path TEXT          Data path.
  --max-depth INTEGER       Max depth parameter.
  --max-leaf-nodes INTEGER  Max leaf nodes parameter.
```

## Predictions

You can make predictions in two ways:
* Batch predictions - direct calls to retrieve the model and score large files.
  * mlflow.sklearn.load_model()
  * mlflow.pyfunc.load_model()
  * Spark UDF - either the DataFrame API or SQL
* Real-time predictions - use MLflow's scoring server to score individual requests.

### Model URI

To retrieve a model from MLflow you need to specify a "model URI" which has several forms. The most common are:
* Model scheme:
   * `models:/my-registered-model/production` - qualify with stage
   * `models:/my-registered-model/123` - qualify with version number
   * `models:/my-registered-model@my-alias` - qualify with alias (new as of MLflow 2.3.0)
* Run scheme - `runs:/7e674524514846799310c41f10d6b99d/my-model`

See MLflow [Referencing Artifacts](https://mlflow.org/docs/latest/concepts.html#referencing-artifacts) page.

### Batch Predictions

You can predict with either a normal Python script or as a `mlflow run` project.

**Normal Python script**
```
python -um wine_quality.predict \
  --model-uri models:/sklearn/production \
  --flavor sklearn 
```

**mlflow run**

See the [MLproject](MLproject) file.
```
mlflow run . \
  -P model-uri=models:/sklearn/production \
  -P flavor=sklearn \
  --entry-point predict 
```

#### 1. Predict with mlflow.sklearn.load_model()

You can use either a `runs` or `models` scheme.

URI with `runs` scheme.
```
python -um wine_quality.predict --model-uri models:/my-registered-model/production --flavor sklearn 

```

URI with `models` scheme.
Assume you have a registered model with a `production` stage or version `1`.
```
python -um wine_quality.predict --model-uri models:/sklearn_wine/production --flavor sklearn 
python -um wine_quality.predict --model-uri models:/sklearn_wine/1 --flavor sklearn 
```

Result.
```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```

Snippet from [predict.py](wine_quality/predict.py):
```
model = mlflow.sklearn.load_model(model_uri)
df = pd.read_csv("../../data/train/wine-quality-white.csv")
predictions = model.predict(data)
```


#### 2. Predict with mlflow.pyfunc.load_model()

```
python -um wine_quality.predict --model-uri models:/my-registered-model/staging --flavor pyfunc
```

```
predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [predict.py](wine_quality/predict.py):
```
data_path = "../../data/train/wine-quality-white.csv"
data = predict_utils.read_prediction_data(data_path)
model_uri = client.get_run(run_id).info.artifact_uri + "/model"
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data)
```

#### 3. Predict with Spark UDF (user-defined function)

See [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf) documentation.

Demonstrates how to invoke a UDF with both the DataFrame API and SQL.

Scroll right to see prediction column.

```
python -um wine_quality.predict \
  --model-uri models:/my-registered-model/production \
  --flavor spark_udf
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
From [predict.py](wine_quality/predict.py):
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

Use the MLflow scoring server to score over HTTP.

#### Scoring Server modes

There are several ways to launch the server:
  1. Local MLflow scoring web server 
  2. Plain docker container
  3. SageMaker docker container
  4. Azure docker container

See MLflow documentation:
* [Built-In Deployment Tools](https://mlflow.org/docs/latest/models.html#built-in-deployment-tools)
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)

#### Input formats

The MLflow scoring server supports the following input formats:
* JSON-serialized Pandas DataFrames
  * Split orientation - [score/wine-quality/wine-quality-split-orient.json](../../data/score/wine-quality/wine-quality-split-orient.json)
  * Records orientation - [score/wine-quality/wine-quality-records-orient.json](../../data/score/wine-quality/wine-quality-records-orient.json)
* CSV

JSON split orientation
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

JSON records orientation
```
[
  {
  "fixed acidity": 7,
  "volatile acidity":  0.27,
  "citric acid": 0.36,
  "residual sugar": 20.7,
  "chlorides": 0.045,
  "free sulfur dioxide": 45,
  "total sulfur dioxide": 170,
  "density": 1.001,
  "pH": 3,
  "sulphates": 0.45,
  "alcohol": 8.8
}
]
```

See MLflow documentation:
* [Deploy MLflow models](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models) - input format documentation
* [pandas.DataFrame.to_json](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html)

#### Scoring

In one window launch the scoring server.

In another window, score some data.
```
curl -X POST -H "Content-Type:application/json" \
  -d @../../data/score/wine-quality/wine-quality-split-orient.json \
  http://localhost:5001/invocations
```
```
[
  [5.470588235294118,5.470588235294118,5.769607843137255]
]
```

#### 1. MLflow scoring web server

Launch the scoring server.
```
mlflow pyfunc serve -port 5001 \
  -model-uri models:/my-registered-model/production
```

Make predictions with curl as described above.

#### 2. Plain Docker Container

See [build-docker](https://mlflow.org/docs/latest/cli.html#mlflow-models-build-docker) documentation.

First build the docker image.
```
mlflow models build-docker \
  --model-uri models:/my-registered-model/production \
  --name dk-wine-sklearn
```

Then launch the scoring server as a docker container.
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
  --model-uri models:/my-registered-model/production \
  --port 5001 --image sm-wine-sklearn
```

You can also launch a scoring server with an ONNX model.
```
mlflow sagemaker run-local \
  --model-uri models:/my-registered-model/production \
  --port 5001 --image sm-wine-sklearn
```

Make predictions with curl as described above.

## Tests

```
cd tests
py.test -s -v test*.py
```
