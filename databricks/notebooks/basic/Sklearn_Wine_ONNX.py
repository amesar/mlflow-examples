# Databricks notebook source
# MAGIC %md # Sklearn MLflow train and predict with ONNX
# MAGIC
# MAGIC **Overview**
# MAGIC * Trains and saves model as Sklearn and ONNX flavor
# MAGIC * Predicts using ONNX native, ONNX PyFunc and ONNX UDF flavors
# MAGIC
# MAGIC **Note**
# MAGIC * Fails with latest `onnx==1.13.1` with: `ImportError: cannot import name 'builder' from 'google.protobuf.internal' `
# MAGIC * Works wth `onnx==1.12.0`

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# %pip install onnx==1.13.1 and 1.14.0 # FAILS with DBR MML 12.2 because of: ImportError: cannot import name 'builder' from 'google.protobuf.internal' 
%pip install onnx==1.12.0
%pip install onnxruntime==1.14.1
%pip install skl2onnx==1.14.0

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

# %run ./Load_Test_Datasources

# COMMAND ----------

dbutils.widgets.text("1. Experiment name", "")
dbutils.widgets.text("2. Registered model", "")
dbutils.widgets.text("3. Max Depth", "1") 
dbutils.widgets.text("4. Max Leaf Nodes", "")

experiment_name = dbutils.widgets.get("1. Experiment name")
model_name = dbutils.widgets.get("2. Registered model")
max_depth = to_int(dbutils.widgets.get("3. Max Depth"))
max_leaf_nodes = to_int(dbutils.widgets.get("4. Max Leaf Nodes"))

model_name = model_name or None
experiment_name = experiment_name or None

print("experiment_name:", experiment_name)
print("model_name:", model_name)
print("max_depth:", max_depth)
print("max_leaf_nodes:", max_leaf_nodes)

# COMMAND ----------

if experiment_name:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print("Experiment:", exp.experiment_id, exp.name)

# COMMAND ----------

import mlflow
import onnx
import skl2onnx

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = WineQuality.load_pandas_data()
train_x, test_x, train_y, test_y = WineQuality.prep_training_data(data)
display(data)

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models.signature import infer_signature

# COMMAND ----------

with mlflow.start_run(run_name="sklearn_onnx") as run:
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

    predictions = sklearn_model.predict(test_x)       
    signature = infer_signature(train_x, predictions) 

    # Log Sklearn model
    mlflow.sklearn.log_model(sklearn_model, "sklearn-model",  
        signature = signature, 
        input_example = test_x
    )
    
    # Log ONNX model
    initial_type = [('float_input', skl2onnx.common.data_types.FloatTensorType([None, test_x.shape[1]]))]
    onnx_model = skl2onnx.convert_sklearn(sklearn_model, initial_types=initial_type)
    print("onnx_model.type:", type(onnx_model))
    mlflow.onnx.log_model(onnx_model, "onnx-model", 
        signature = signature, 
        input_example = test_x,
        registered_model_name=model_name
    )
        
    # Run predictions and log metrics
    predictions = sklearn_model.predict(test_x)
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(test_y, predictions)))

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/onnx-model"
model_uri

# COMMAND ----------

data_to_predict = WineQuality.prep_prediction_data(data)

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
