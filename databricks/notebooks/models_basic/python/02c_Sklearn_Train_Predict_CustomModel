# Databricks notebook source
# MAGIC %md # Basic Sklearn MLflow train and predict with Custom Model
# MAGIC * Demonstrate the use of MLflow [Python custom models](https://mlflow.org/docs/latest/models.html#custom-python-models).
# MAGIC * Variant of [02a_Sklearn_Train_Predict]($02a_Sklearn_Train_Predict).
# MAGIC * Three custom models:
# MAGIC   * CustomProbaModel - custom call to DecisionTreeClassifier.predict_proba() instead of default Pyfunc call to DecisionTreeClassifier.predict().
# MAGIC   * CustomResponseModel - return a custom response (dict) for [Pyfunc.predict](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.predict) response instead of standard response (pandas.DataFrame, pandas.Series, numpy.ndarray or list).
# MAGIC   * CustomCodeModel - write you own non-Sklearn code for predictions.

# COMMAND ----------

# MAGIC %md ### Setup

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
print("MLflow Version:", mlflow.__version__)
print("sklearn version:",sklearn.__version__)
print("sparkVersion:", get_notebook_tag("sparkVersion"))

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

# MAGIC %md ### Custom Models

# COMMAND ----------

# Return predict_proba() instead of predict()
class CustomProbaModel(mlflow.pyfunc.PythonModel):
   def __init__(self, model):
       self.model = model
   def predict(self, context, data):
       return self.model.predict_proba(data)
    
# Return a dict instead of Pandas.DataFrame, pandas.Series or numpy.ndarray 
class CustomResponseModel(mlflow.pyfunc.PythonModel):
   def __init__(self, model):
       self.model = model
   def predict(self, context, data):
       predictions =  self.model.predict(data)
       return{ f"{i}":p for i,p in enumerate(predictions) } 
 
# No sklearn model at all - custom prediction code
class CustomCodeModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass
    def predict(self, context, data):
        return [ j for j in range(0, data.shape[0]) ]

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

with mlflow.start_run(run_name="sklearn") as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    print("Parameters:")
    print("  max_depth:",max_depth)
    print("  max_leaf_nodes:",max_leaf_nodes)
    
    mlflow.set_tag("version.mlflow", mlflow.__version__)

    model = DecisionTreeClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
    model.fit(train_x, train_y)
      
    predictions = model.predict(test_x)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        
    # Log sklearn model
    mlflow.sklearn.log_model(model, "sklearn-model")
    
    # Log custom Pyfunc models
    mlflow.pyfunc.log_model("pyfunc-custom-proba-model", python_model=CustomProbaModel(model))
    mlflow.pyfunc.log_model("pyfunc-custom-response-model", python_model=CustomResponseModel(model))
    mlflow.pyfunc.log_model("pyfunc-custom-code-model", python_model=CustomCodeModel())
    
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

# MAGIC %md #### Predict as standard sklearn

# COMMAND ----------

model_uri = f"runs:/{run_id}/sklearn-model"
model_uri

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
data_to_predict = data.drop(colLabel, axis=1)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as Pyfunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions,columns=[colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as CustomProbaModel

# COMMAND ----------

model_uri = f"runs:/{run_id}/pyfunc-custom-proba-model"
model_uri

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
type(predictions), predictions.shape

# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md #### Predict as CustomResponseModel

# COMMAND ----------

model_uri = f"runs:/{run_id}/pyfunc-custom-response-model"
model_uri

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
type(predictions)

# COMMAND ----------

predictions

# COMMAND ----------

# MAGIC %md #### Predict as CustomCodeModel

# COMMAND ----------

model_uri = f"runs:/{run_id}/pyfunc-custom-code-model"
model_uri

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
type(predictions)

# COMMAND ----------

predictions