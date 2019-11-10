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
  *  UDF
* Data: [../data/wine-quality-white.csv](../data/wine-quality-white.csv)

## Training

Source: [main.py](main.py) and [wine_quality/train.py](wine_quality/train.py).

### Unmanaged without mlflow run

#### Command-line python

To run with standard main function
```
python main.py --experiment_name sklearn \
  --max_depth 2 --max_leaf_nodes 32
```

### Using mlflow run

These runs use the [MLproject](MLproject) file. For more details see [MLflow documentation - Running Projects](https://mlflow.org/docs/latest/projects.html#running-projects).

Note that mlflow run ignores the `set_experiment()` function so you must specify the experiment with the  `--experiment-sklearn` argument.

**mlflow run local**
```
mlflow run . \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=LocalRun \
  --experiment-name=sklearn
```

**mlflow run github**
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=GitRun \
  --experiment-name=sklearn
```

**mlflow run Databricks remote** - Run against Databricks. 

See [Remote Execution on Databricks](https://mlflow.org/docs/latest/projects.html#remote-execution-on-databricks) and [mlflow_run_cluster.json](mlflow_run_cluster.json).

Setup.
```
export MLFLOW_TRACKING_URI=databricks
```
The token and tracking server URL will be picked up from your Databricks CLI ~/.databrickscfg default profile.

Now run.
```
mlflow run https://github.com/amesar/mlflow-fun.git#examples/scikit-learn/wine-quality \
  -P max_depth=2 -P max_leaf_nodes=32 -P run_origin=GitRun \
  -P data_path=/dbfs/tmp/data/wine-quality-white.csv \
  --experiment-name=sklearn \
  --backend databricks --backend-config mlflow_run_cluster.json
```

### Databricks Cluster Runs

You can also package your code as an wheel and run it with the standard Databricks REST API endpoints
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
databricks fs cp main.py dbfs:/tmp/jobs/wine_quality/main.py
databricks fs cp data/wine-quality-white.csv dbfs:/tmp/jobs/wine_quality/wine-quality-white.csv
databricks fs cp \
  dist/mlflow_fun-0.0.1-py3-none-any.whl \
  dbfs:/tmp/jobs/wine_quality/mlflow_fun-0.0.1-py3-none-any.whl
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
  --python-params '[ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv" ]'
```

##### Run with existing cluster
First create a job with the spec file [create_job_existing_cluster.json](create_job_existing_cluster.json).
```
databricks jobs create --json-file create_job_existing_cluster.json
```

Then run the job with desired parameters.
```
databricks jobs run-now --job-id $JOB_ID --python-params ' [ "WineQualityExperiment", 0.3, 0.3, "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv" ] '
```


#### Run wheel from Databricks notebook

Create a notebook with the following cell. Attach it to the existing cluster described above.
```
from wine_quality import Trainer
data_path = "/dbfs/tmp/jobs/wine_quality/wine-quality-white.csv"
trainer = Trainer("WineQualityExperiment", data_path, "from_notebook_with_wheel")
trainer.train(0.4, 0.4)
```

## Predictions

You can make predictions in the following ways:
1. Use MLflow's serving web server and submit predictions via HTTP calls
2. Call mlflow.sklearn.load_model() from your own serving code and then make predictions
4. Call mlflow.pyfunc.load_pyfunc() from your own serving code and then make predictions
5. Batch prediction with Spark UDF (user-defined function)


See MLflow documentation:
* [Tutorial - Serving the Model](https://www.mlflow.org/docs/latest/tutorial.html#serving-the-model)
* [Quickstart - Saving and Serving Models](https://www.mlflow.org/docs/latest/quickstart.html#saving-and-serving-models)
* [mlflow.pyfunc.spark_udf](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf)


### Data for predictions
[../data/predict-wine-quality.json](../data/predict-wine-quality.json)
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
  "data": [ [ 12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66 ] ]
}
```

### 1. Serving Models from MLflow Web Server

In one window run the server.
```
mlflow pyfunc serve -p 5001 -r 7e674524514846799310c41f10d6b99d -m model
```

In another window, submit a prediction.
```
curl -X POST -H "Content-Type:application/json" -d @data/wine-quality-red.csv http://localhost:5001/invocations

[
    5.551096337521979,
    5.297727513113797,
    5.427572126267637,
    5.562886443251915,
    5.562886443251915
]
```

### 2. Predict with mlflow.sklearn.load_model()

```
python sklearn_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [sklearn_predict.py](sklearn_predict.py):
```
model = mlflow.sklearn.load_model("model",run_id="7e674524514846799310c41f10d6b99d")
df = pd.read_csv("data/wine-quality-red.csv")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 3. Predict with mlflow.pyfunc.load_pyfunc()

```
python pyfunc_predict.py 7e674524514846799310c41f10d6b99d

predictions: [5.55109634 5.29772751 5.42757213 5.56288644 5.56288644]
```
From [pyfunc_predict.py](pyfunc_predict.py):
```
model_uri = mlflow.start_run("7e674524514846799310c41f10d6b99d").info.artifact_uri +  "/model"
model = mlflow.pyfunc.load_pyfunc(model_uri)
df = pd.read_csv("data/wine-quality-red.csv")
predicted = model.predict(df)
print("predicted:",predicted)
```

### 4. Batch prediction with Spark UDF (user-defined function)

Scroll right to see prediction column.

```
pip install pyarrow

spark-submit --master local[2] spark_udf_predict.py 7e674524514846799310c41f10d6b99d

+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|alcohol|chlorides|citric acid|density|fixed acidity|free sulfur dioxide|  pH|residual sugar|sulphates|total sulfur dioxide|volatile acidity|        prediction|
+-------+---------+-----------+-------+-------------+-------------------+----+--------------+---------+--------------------+----------------+------------------+
|    8.8|    0.045|       0.36|  1.001|          7.0|               45.0| 3.0|          20.7|     0.45|               170.0|            0.27| 5.551096337521979|
|    9.5|    0.049|       0.34|  0.994|          6.3|               14.0| 3.3|           1.6|     0.49|               132.0|             0.3| 5.297727513113797|
|   10.1|     0.05|        0.4| 0.9951|          8.1|               30.0|3.26|           6.9|     0.44|                97.0|            0.28| 5.427572126267637|
|    9.9|    0.058|       0.32| 0.9956|          7.2|               47.0|3.19|           8.5|      0.4|               186.0|            0.23| 5.562886443251915|
```
From [spark_udf_predict.py](spark_udf_predict.py):
```
spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
df = spark.read.option("inferSchema",True).option("header", True).csv("data/wine-quality-red.csv")
df = df.drop("quality")

udf = mlflow.pyfunc.spark_udf(spark, "model", run_id="7e674524514846799310c41f10d6b99d")
df2 = df.withColumn("prediction", udf(*df.columns))
df2.show(10)
```
