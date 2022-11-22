# Databricks notebook source
# MAGIC %md # SparkML MLflow train and predict notebook
# MAGIC * Trains and saves model as Spark ML
# MAGIC * Predicts as Spark ML
# MAGIC * Spark.autolog - automatically logs datasource in tag `sparkDatasourceInfo`
# MAGIC   * https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.autolog
# MAGIC   * [Sample run with tag sparkDatasourceInfo](https://demo.cloud.databricks.com/#mlflow/experiments/6682638/runs/5527319e739a450b9af24b1dc98e1c59)

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("Max Depth", "5")
dbutils.widgets.text("Max Bins", "32")
dbutils.widgets.dropdown("UDF predict","no",["yes","no"])
dbutils.widgets.text("Registered Model","")

maxDepth = int(dbutils.widgets.get("Max Depth"))
maxBins = float(dbutils.widgets.get("Max Bins"))
udf_predict = dbutils.widgets.get("UDF predict") == "yes"
registered_model = dbutils.widgets.get("Registered Model")
if registered_model=="": registered_model = None

maxDepth, maxBins, udf_predict, registered_model

# COMMAND ----------

import mlflow
import mlflow.spark
import pyspark
print("MLflow Version:", mlflow.__version__)
print("Spark Version:", spark.version)
print("PySpark Version:", pyspark.__version__)
print("sparkVersion:", get_notebook_tag("sparkVersion"))

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

data = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load(data_path.replace("/dbfs","dbfs:")) 
(trainData, testData) = data.randomSplit([0.7, 0.3], 42)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

with mlflow.start_run() as run:
    run_id = run.info.run_uuid
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
    
    # Create model
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, \
                               maxDepth=maxDepth, maxBins=maxBins)

    # Create pipeline
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model
    model = pipeline.fit(trainData)
    
    spark_model_name = "model"
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)
    
    # Log MLflow training metrics
    metrics = ["rmse","r2", "mae"]
    print("Metrics:")
    predictions = model.transform(testData)
    
    for metric in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric)
        v = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric,v))
        mlflow.log_metric(metric,v)
        
    mlflow.spark.log_model(model, spark_model_name, registered_model_name=registered_model)
    print("Model:",spark_model_name)

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/{spark_model_name}"
#model_uri = f"models:/andre_03a_SparkML_Train_Predict/production"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as Spark ML

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)

# COMMAND ----------

predictions = model.transform(data)
display(predictions.select(colPrediction, colLabel, colFeatures))

# COMMAND ----------

type(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as PyFunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.predict(data.toPandas())
type(predictions),len(predictions)

# COMMAND ----------

display(pd.DataFrame(predictions))

# COMMAND ----------

# MAGIC %md #### Predict as Spark UDF
# MAGIC 
# MAGIC Error:
# MAGIC ```
# MAGIC py4j.protocol.Py4JJavaError: An error occurred while calling o55.transform.
# MAGIC : java.lang.IllegalArgumentException: Field "fixed acidity" does not exist.
# MAGIC Available fields: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
# MAGIC ```

# COMMAND ----------

if udf_predict:
    model_uri = f"runs:/{run_id}/spark-model"
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = data.withColumn(colPrediction, udf(*data.columns))
    display(predictions)