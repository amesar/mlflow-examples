# Databricks notebook source
# MAGIC %md ## MLflow Python train and Scala score
# MAGIC
# MAGIC Train and log an MLflow Spark ML model in Python, and then read and score it in Scala.
# MAGIC
# MAGIC **Overview**
# MAGIC
# MAGIC Python: 
# MAGIC * Train and log model as MLflow Spark ML flavor.
# MAGIC * Score the model.
# MAGIC
# MAGIC Scala: 
# MAGIC * Read the Spark ML model artifact with the Java [downloadArtifacts()](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowClient.html#downloadArtifacts-java.lang.String-java.lang.String-) method.
# MAGIC * Score the model.
# MAGIC * See Scala scoring section for details.
# MAGIC
# MAGIC Last updated: 2023-07-03

# COMMAND ----------

# MAGIC %md ### Python

# COMMAND ----------

# MAGIC %run ./Common_Python

# COMMAND ----------

# MAGIC %md #### Get data

# COMMAND ----------

data = WineQuality.load_pandas_data()
data = spark.createDataFrame(data)
(trainData, testData) = data.randomSplit([0.7, 0.3], 42)

# COMMAND ----------

# MAGIC %md #### Train

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow.spark

maxDepth = 5
maxBins = 32

with mlflow.start_run() as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print("run_id:",run_id)
    print("experiment_id:",experiment_id) 
    
    # Set MLflow tags
    mlflow.set_tag("mlflow_version", mlflow.__version__)
    mlflow.set_tag("spark_version", spark.version)
    mlflow.set_tag("pyspark_version", pyspark.__version__)
    mlflow.set_tag("sparkVersion", get_notebook_tag("sparkVersion"))
    mlflow.set_tag("DATABRICKS_RUNTIME_VERSION", os.environ.get('DATABRICKS_RUNTIME_VERSION'))
    
    # Log MLflow parameters
    print("Parameters:")
    print("  maxDepth:",maxDepth)
    print("  maxBins:",maxBins)
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)
    
    # Create model
    model = DecisionTreeRegressor(labelCol=WineQuality.colLabel, 
        featuresCol=WineQuality.colFeatures, \
        maxDepth=maxDepth, maxBins=maxBins)

    # Create pipeline
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=WineQuality.colFeatures)
    pipeline = Pipeline(stages=[assembler, model])
    
    # Fit model
    model = pipeline.fit(trainData)
    
    # Log MLflow training metrics
    print("Metrics:")
    predictions = model.transform(testData)
    metrics = ["rmse", "r2", "mae"]
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=WineQuality.colLabel, predictionCol=WineQuality.colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        print(f"  {metric}: {v}")
        mlflow.log_metric(metric,v)
    
    # Log MLflow model
    mlflow.spark.log_model(model, "model")

# COMMAND ----------

# MAGIC %md #### Display run UI link

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

# MAGIC %md #### Save run_id for Scala scoring

# COMMAND ----------

spark.conf.set("RUN_ID",run_id)

# COMMAND ----------

# MAGIC %md #### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
model_uri

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.transform(data)
display(predictions.select(WineQuality.colPrediction, WineQuality.colLabel, WineQuality.colFeatures))

# COMMAND ----------

# MAGIC %md ### Scala - predict model created by Python

# COMMAND ----------

# MAGIC %scala 
# MAGIC import org.mlflow.tracking.MlflowClient
# MAGIC val client = new MlflowClient()

# COMMAND ----------

# MAGIC %md #### Get runId and dataPath from Python

# COMMAND ----------

# MAGIC %scala
# MAGIC val runId = spark.conf.get("RUN_ID")

# COMMAND ----------

# MAGIC %md #### Prepare data

# COMMAND ----------

# MAGIC %scala
# MAGIC val data = getData()
# MAGIC val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 42)

# COMMAND ----------

# MAGIC %md #### Load model from Python run
# MAGIC * The Java MLflow client doesn't have a concept of MLflow model flavors like Python does.
# MAGIC * Therefore the Java MLflow client doesn't have a `loadModel` method analagous to Python [load_model](https://mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.load_model).
# MAGIC * To load the model, you have to use the lower-level Java [downloadArtifacts](https://mlflow.org/docs/latest/java_api/org/mlflow/tracking/MlflowClient.html#downloadArtifacts-java.lang.String-java.lang.String-) method.
# MAGIC * You simply have to tweak the model's artifact path to conform to the Spark ML flavor artifact convention.

# COMMAND ----------

# MAGIC %scala
# MAGIC val localPath = client.downloadArtifacts(runId, "model/sparkml").getAbsolutePath

# COMMAND ----------

# MAGIC %md ##### Since downloadArtifacts returns a local path, we need to copy it to DBFS to use SparkML

# COMMAND ----------

# MAGIC %scala
# MAGIC import java.io.File
# MAGIC import org.apache.commons.io.FileUtils
# MAGIC val dbfsPath = s"/dbfs/tmp/$runId"
# MAGIC FileUtils.copyDirectory(new File(localPath), new File(dbfsPath))

# COMMAND ----------

# MAGIC %md ##### Load model

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.PipelineModel
# MAGIC val model = PipelineModel.load(dbfsPath.replace("/dbfs","dbfs:"))

# COMMAND ----------

# MAGIC %md #### Predict

# COMMAND ----------

# MAGIC %scala
# MAGIC val predictions = model.transform(data)
# MAGIC display(predictions.select("prediction", "quality", "features"))

# COMMAND ----------

# MAGIC %md ### Result - run_id

# COMMAND ----------

run_id
