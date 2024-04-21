# Databricks notebook source
# MAGIC %md # SparkML MLflow train and predict notebook
# MAGIC * Trains and saves model as Spark ML
# MAGIC * Predicts as Spark ML
# MAGIC * Spark.autolog - automatically logs datasource in tag `sparkDatasourceInfo`
# MAGIC   * https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.autolog
# MAGIC   * [Sample run with tag sparkDatasourceInfo](https://demo.cloud.databricks.com/#mlflow/experiments/6682638/runs/5527319e739a450b9af24b1dc98e1c59)

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Registered Model","")
dbutils.widgets.text("2. Delta table", "")
dbutils.widgets.dropdown("3. UDF predict","no",["yes","no"])
dbutils.widgets.dropdown("4. Log input", "no", ["yes","no"])
dbutils.widgets.text("5. Max Depth", "5")
dbutils.widgets.text("6. Max Bins", "32")

registered_model = dbutils.widgets.get("1. Registered Model")
delta_table = dbutils.widgets.get("2. Delta table")
udf_predict = dbutils.widgets.get("3. UDF predict") == "yes"
log_input = dbutils.widgets.get("4. Log input") == "yes"
maxDepth = int(dbutils.widgets.get("5. Max Depth"))
maxBins = float(dbutils.widgets.get("6. Max Bins"))

if registered_model=="": registered_model = None

set_model_registry(registered_model)

print("\nregistered_model:", registered_model)
print("delta_table:", delta_table)
print("udf_predict:", udf_predict)
print("log_input:", log_input)
print("maxDepth:", maxDepth)
print("maxBins:", maxBins) 

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

data, data_source = WineQuality.get_data(delta_table)
data_source

# COMMAND ----------

display(data)

# COMMAND ----------

(X_train, X_test) = data.randomSplit([0.7, 0.3], 42)

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
    model = DecisionTreeRegressor(
        labelCol=WineQuality.colLabel, 
        featuresCol=WineQuality.colFeatures,
         maxDepth=maxDepth, maxBins=maxBins)
    mlflow.set_tag("algorithm", type(model))

    # Create pipeline
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=WineQuality.colFeatures)
    pipeline = Pipeline(stages=[assembler, model])
    
    # Fit model
    model = pipeline.fit(X_train)
    
    spark_model_name = "model"
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)
    
    # Log MLflow training metrics
    metrics = ["rmse","r2", "mae"]
    print("Metrics:")
    predictions = model.transform(X_test)
    
    for metric in metrics:
        evaluator = RegressionEvaluator(
            labelCol = WineQuality.colLabel, 
            predictionCol = WineQuality.colPrediction, 
            metricName = metric
        )
        v = evaluator.evaluate(predictions)
        print(f"  {metric}: {v}")
        mlflow.log_metric(metric, v)
        
    mlflow.spark.log_model(model, spark_model_name, registered_model_name=registered_model)
    print("Model:", spark_model_name)

    log_data_input(run, log_input, data_source, X_train)

# COMMAND ----------

display_run_uri(experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Show input data

# COMMAND ----------

run = client.get_run(run_id)
if hasattr(run, "inputs") and run.inputs:
    for input in run.inputs:
        print(input)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/{spark_model_name}"
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as Spark ML

# COMMAND ----------

model = mlflow.spark.load_model(model_uri)
type(model)

# COMMAND ----------

predictions = model.transform(data)
display(predictions.select(WineQuality.colPrediction, WineQuality.colLabel, WineQuality.colFeatures))

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
# MAGIC
# MAGIC Error:
# MAGIC ```
# MAGIC 404 Client Error: Not Found for url: https://e2-demo-west-root.s3.us-west-2.amazonaws.com/oregon-prod/2556758628403379.jobs/mlflow-tracking/959d3635684d48bab13a0dbe5c1ea298/d4c9187f7fd94c13ab2c5a0ff74e3129/artifacts/spark-model?
# MAGIC ```

# COMMAND ----------

if udf_predict:
    model_uri = f"runs:/{run_id}/spark-model"
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = data.withColumn(WineQuality.colPrediction, udf(*data.columns))
    display(predictions)
