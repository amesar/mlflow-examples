# Databricks notebook source
# MAGIC %md # Sklearn MLflow train and predict for CICD
# MAGIC * Trains and saves model as sklearn
# MAGIC * Predicts using sklearn and pyfunc UDF flavors
# MAGIC * https://demo.cloud.databricks.com/#mlflow/experiments/ID6230897

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

import sklearn
import mlflow
import mlflow.sklearn
print("MLflow Version:", mlflow.__version__)
print("sklearn version:",sklearn.__version__)

# COMMAND ----------

default_experiment_name = "/Users/andre.mesarovic@databricks.com/tmp/02a_Sklearn_Train_Predict"
dbutils.widgets.text("Experiment Name", default_experiment_name) 
dbutils.widgets.text("Max Depth", "1") 
dbutils.widgets.text("Max Leaf Nodes", "32")
dbutils.widgets.text("Run Name", "Local")

experiment_name = dbutils.widgets.get("Experiment Name")
max_depth = int(dbutils.widgets.get("Max Depth"))
max_leaf_nodes = int(dbutils.widgets.get("Max Leaf Nodes"))
run_name = dbutils.widgets.get("Run Name")
experiment_name, run_name, max_depth, max_leaf_nodes

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
mlflow.set_experiment(experiment_name)
experiment = client.get_experiment_by_name(experiment_name)
experiment.experiment_id, experiment_name

# COMMAND ----------

colLabel = "quality"
colPrediction = "prediction"
colFeatures = "features"

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data_path = download_wine_file()

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=2019)
train_x = train.drop([colLabel], axis=1)
test_x = test.drop([colLabel], axis=1)
train_y = train[colLabel]
test_y = test[colLabel]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    print("Parameters:")
    print("  max_depth:",max_depth)
    print("  max_leaf_nodes:",max_leaf_nodes)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

    model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
    mlflow.sklearn.log_model(model, "sklearn-model")
    
    predictions = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 

# COMMAND ----------

#display_run_uri(run.info.experiment_id, run_id)
# https://demo.cloud.databricks.com/#mlflow/experiments/6230591/runs/e5e4879ed7c447ffaa72dc7fec0833c6

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = "runs:/{}/sklearn-model".format(run_id)

# COMMAND ----------

# MAGIC %md #### Predict as sklearn

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns))
display(predictions)