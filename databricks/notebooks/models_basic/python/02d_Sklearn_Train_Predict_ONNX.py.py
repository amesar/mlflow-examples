# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict with ONNX
# MAGIC * Trains and saves model as sklearn and ONNX
# MAGIC * Predicts using ONNX native, ONNX PyFunc and ONNX UDF flavors

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %pip install onnx==1.13.0
# MAGIC %pip install onnxruntime==1.13.1
# MAGIC %pip install skl2onnx

# COMMAND ----------

# MAGIC %run ./common

# COMMAND ----------

dbutils.widgets.text("Max Depth", "1") 
dbutils.widgets.text("Max Leaf Nodes", "")
max_depth = to_int(dbutils.widgets.get("Max Depth"))
max_leaf_nodes = to_int(dbutils.widgets.get("Max Leaf Nodes"))

max_depth, max_leaf_nodes

# COMMAND ----------

import sklearn
import mlflow
import mlflow.sklearn
import mlflow.onnx
import onnx

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = get_wine_quality_data()
display(data)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)

# The predicted column is colLabel which is a scalar from [3, 9]
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

with mlflow.start_run(run_name="sklearn") as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    
    mlflow.set_tag("version.mlflow", mlflow.__version__)
    mlflow.set_tag("version.onnx", onnx.__version__)

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

    sklearn_model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    sklearn_model.fit(train_x, train_y)
             
    # Log Sklearn model
    mlflow.sklearn.log_model(sklearn_model, "sklearn-model")
    
    # Log ONNX model
    initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, test_x.shape[1]]))]
    onnx_model = skl2onnx.convert_sklearn(sklearn_model, initial_types=initial_type)
    print("onnx_model.type:", type(onnx_model))
    mlflow.onnx.log_model(onnx_model, "onnx-model")
        
    # Run predictions and log metrics
    predictions = sklearn_model.predict(test_x)
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(test_y, predictions)))

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict as ONNX

# COMMAND ----------

#model_uri = f"runs:/{run_id}/onnx-model"
#model_uri

# COMMAND ----------

# MAGIC %md ### Predict ONNX

# COMMAND ----------

model_uri = f"runs:/{run_id}/onnx-model"
model_uri

# COMMAND ----------

data_to_predict = data.drop(colLabel, axis=1)

# COMMAND ----------

# MAGIC %md #### Predict as PyFunc ONNX flavor

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
type(predictions), predictions.shape

# COMMAND ----------

display(predictions)

# COMMAND ----------

# MAGIC %md #### Predict as native ONNX flavor

# COMMAND ----------

import mlflow.onnx
import onnxruntime
import numpy as np

model = mlflow.onnx.load_model(model_uri)
session = onnxruntime.InferenceSession(model.SerializeToString())
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: data_to_predict.to_numpy().astype(np.float32)})[0]
type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=["prediction"]))

# COMMAND ----------

# MAGIC %md #### Predict as ONNX UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn("prediction", udf(*df.columns)).select("prediction")
display(predictions)