# Databricks notebook source
# MAGIC %md # MLflow Spark ML Training Tutorial
# MAGIC 
# MAGIC **Overview**
# MAGIC * Train a SparkML model several times with different `maxDepth` hyperparameters
# MAGIC * Algorithm is DecisionTreeRegressor with wine quality dataset
# MAGIC * Show different ways to view runs:
# MAGIC   * [MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos)
# MAGIC   * [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs)
# MAGIC   * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs)
# MAGIC   * Experiment data source
# MAGIC * Find the best run for the experiment
# MAGIC * Show how to score the model with different flavors;
# MAGIC   * [Spark flavor](https://mlflow.org/docs/latest/python_api/mlflow.spark.html) 
# MAGIC   * [Pyfunc flavor](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) 
# MAGIC   * [MLeap flavor](https://mlflow.org/docs/latest/python_api/mlflow.mleap.html) (using SparkBundle)
# MAGIC   
# MAGIC **MLeap**
# MAGIC * MLeap: common serialization format and execution engine for machine learning pipelines
# MAGIC * https://mleap-docs.combust.ml
# MAGIC * Databricks MLeap documentation:
# MAGIC   * [MLeap ML Model Export](https://docs.databricks.com/applications/machine-learning/model-export-import/mleap-model-export.html#mleap-ml-model-export)
# MAGIC   * [Train a PySpark model and save in MLeap format](https://docs.databricks.com/applications/mlflow/tracking-ex-pyspark.html#train-a-pyspark-model-and-save-in-mleap-format) - Databricks documentation notebook

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

experiment_id, experiment_name = init()

# COMMAND ----------

# MAGIC %md ### Delete any existing runs

# COMMAND ----------

delete_runs(experiment_id)

# COMMAND ----------

# MAGIC %md ## Prepare data

# COMMAND ----------

data_path = download_data()

# COMMAND ----------

data = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load(data_path.replace("/dbfs","dbfs:")) 
(trainData, testData) = data.randomSplit([0.7, 0.3], 42)

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md ### Training Pipeline

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

import mlflow.spark
import mlflow.mleap

def train(maxDepth):
    with mlflow.start_run() as run:        
        # Set MLflow tags
        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.set_tag("spark_version", spark.version)
        mlflow.set_tag("pyspark_version", pyspark.__version__)
        
        # Log MLflow parameters
        mlflow.log_param("maxDepth", maxDepth)
        
        # Create model
        dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, \
                                   maxDepth=maxDepth)
    
        # Create pipeline
        assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
        pipeline = Pipeline(stages=[assembler, dt])
        
        # Fit model
        model = pipeline.fit(trainData)
        
        # Predict on test data
        predictions = model.transform(testData)
        
        # Log MLflow training metrics
        predictions = model.transform(testData)
        metric = "rmse"
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        mlflow.log_metric(metric, v)
        print(f"{v:5.3f} {maxDepth:2d} {run.info.run_id} {run.info.experiment_id}")
        
        # Log MLflow model as Spark ML
        mlflow.spark.log_model(model, "spark-model")
        
        # Log MLflow model as MLeap
        mlflow.mleap.log_model(spark_model=model, sample_input=testData, artifact_path="mleap-model")

# COMMAND ----------

# MAGIC %md ### Train with different hyperparameters

# COMMAND ----------

params = [1, 2, 4, 16]
for p in params:
    train(p)

# COMMAND ----------

# MAGIC %md ### Different ways to show an experiment's runs

# COMMAND ----------

# MAGIC %md #### MlflowClient.list_run_infos
# MAGIC * [mlflow.tracking.MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos)
# MAGIC * Returns a list of [RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) objects

# COMMAND ----------

infos = client.list_run_infos(experiment_id)
for info in infos:
    print(info.run_id, info.experiment_id, info.status)

# COMMAND ----------

# MAGIC %md #### MLflowClient.search_runs
# MAGIC * [mlflow.tracking.MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs)
# MAGIC * Returns a list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects
# MAGIC * Allows for paging when you have a very large number of runs
# MAGIC * Sorted by best metrics `rmse`

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"])
for run in runs:
    print(run.info.run_id, run.data.metrics["rmse"], run.data.params)

# COMMAND ----------

# MAGIC %md #### mlflow.search_runs
# MAGIC * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs)
# MAGIC * Returns a Pandas dataframe
# MAGIC * All `data` attributes are exploded into one flat column name space
# MAGIC * Sorted by best metrics `rmse`

# COMMAND ----------

runs = mlflow.search_runs(experiment_id)
runs = runs.sort_values(by=['metrics.rmse'])
runs

# COMMAND ----------

runs[["run_id","metrics.rmse","params.maxDepth"]]

# COMMAND ----------

# MAGIC %md #### Experiment data source
# MAGIC * Returns a Spark dataframe of all runs
# MAGIC * Run `data` elements such as `params`, `metrics` and `tags` are nested.
# MAGIC * Background Documentation:
# MAGIC   * Databricks documentation:
# MAGIC     * [MLflow Experiment Data Source](https://docs.databricks.com/data/data-sources/mlflow-experiment.html#mlflow-exp-datasource)
# MAGIC     * [Analyze MLflow runs using DataFrames
# MAGIC ](https://docs.databricks.com/applications/mlflow/tracking.html#analyze-mlflow-runs-using-dataframes)
# MAGIC   * [Analyzing Your MLflow Data with DataFrames](https://databricks.com/blog/2019/10/03/analyzing-your-mlflow-data-with-dataframes.html) - blog - 2019-10-03

# COMMAND ----------

from pyspark.sql.functions import *
df_runs = spark.read.format("mlflow-experiment").load(experiment_id)
df_runs.createOrReplaceTempView("runs")

# COMMAND ----------

# MAGIC %md ##### Query with Spark DataFrame API

# COMMAND ----------

df_runs = df_runs.sort(asc("metrics.rmse"))
display(df_runs)

# COMMAND ----------

display(df_runs.select("run_id", round("metrics.rmse",3).alias("rmse"),"params"))

# COMMAND ----------

# MAGIC %md ##### Query as SQL

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc limit 1

# COMMAND ----------

# MAGIC %md ### Find the best run

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
best_run

# COMMAND ----------

display_run_uri(experiment_id, best_run.info.run_id)

# COMMAND ----------

best_run.info.run_id, best_run.data.metrics["rmse"]

# COMMAND ----------

# MAGIC %md ### Score
# MAGIC 
# MAGIC Several ways to score:
# MAGIC * Spark ML flavor
# MAGIC * Pyfunc flavor
# MAGIC * MLeap (SparkBundle) flavor

# COMMAND ----------

model_uri = f"runs:/{best_run.info.run_id}/spark-model"
model_uri

# COMMAND ----------

# MAGIC %md #### Score with Spark ML flavor

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.transform(data)
display(predictions.select(colPrediction, colLabel, colFeatures))

# COMMAND ----------

# MAGIC %md #### Score with Pyfunc flavor

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.predict(data.toPandas())
type(predictions)

# COMMAND ----------

predictions[:10]

# COMMAND ----------

# MAGIC %md #### Score with MLeap flavor 
# MAGIC * Uses SparkBundle
# MAGIC * There is no MLflow MLeap `load_model` method so we have to:
# MAGIC   * Manually construct the model artifact URI
# MAGIC   * Use low-level MLeap methods to load the model

# COMMAND ----------

run = client.get_run(best_run.info.run_id)
run.info.artifact_uri

# COMMAND ----------

bundle_path = f"file:{run.info.artifact_uri}/mleap-model/mleap/model".replace("dbfs:","/dbfs")
bundle_path

# COMMAND ----------

from pyspark.ml import PipelineModel
from mleap.pyspark.spark_support import SimpleSparkSerializer
model = PipelineModel.deserializeFromBundle(bundle_path)
type(model)

# COMMAND ----------

predictions = model.transform(data)
type(predictions)

# COMMAND ----------

display(predictions.select(colPrediction, colLabel, colFeatures))