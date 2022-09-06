# Databricks notebook source
# MAGIC %md # Real-time score model from Model Registry
# MAGIC * Real-time scoring against MLflow model server with curl.
# MAGIC * Scores the best model run created in [01_MLflow_Sklearn_Train_Tutorial]($01_MLflow_Sklearn_Train_Tutorial) and registered in [02_Promote_to_Model_Registry]($02_Model_Registry_Tutorial) notebooks.
# MAGIC * First start the model serving cluster: 
# MAGIC   * Navigate to the https://demo.cloud.databricks.com/#mlflow/models/MLflow_Sklearn_Train_Tutorial/serving cluster page.
# MAGIC   * Click the `Enable Serving` button and wait until cluster is in `Ready Status`.
# MAGIC   * Then proceeed to run this notebook.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

dbutils.widgets.removeAll()


# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("Registered model name", default_model_name)
model_name = dbutils.widgets.get("Registered model name")
model_name

# COMMAND ----------

# MAGIC %md ### Start model serving cluster
# MAGIC Go to the Serving tab of the model's UI page - see link below.

# COMMAND ----------

display_registered_model_uri(model_name)

# COMMAND ----------

# MAGIC %md ### Create model scoring URI and transient token

# COMMAND ----------

import os

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host_name = context.tags().get("browserHostName").get()
scoring_uri = f"https://{host_name}/model/{model_name}/Production/invocations"

os.environ["SCORING_URI"] = scoring_uri
os.environ["TOKEN"] = context.apiToken().get() 
os.environ["HOST_NAME"] = host_name 
os.environ["MODEL_NAME"] = model_name

scoring_uri, model_name

# COMMAND ----------

# MAGIC %sh echo $HOST_NAME

# COMMAND ----------

# MAGIC %md ### List endpoints

# COMMAND ----------

# MAGIC %sh
# MAGIC LIST_URI="https://$HOST_NAME/2.0/preview/mlflow/endpoints/list"
# MAGIC echo $LIST_URI
# MAGIC curl -X GET -H "Authorization: Bearer $TOKEN" $LIST_URI  

# COMMAND ----------

# MAGIC %md ### Start server - JSON

# COMMAND ----------

# MAGIC %sh
# MAGIC ENABLE_URI="https://$HOST_NAME/2.0/preview/mlflow/endpoints/enable"
# MAGIC echo $ENABLE_URI
# MAGIC echo $MODEL_NAME
# MAGIC 
# MAGIC curl -v -H "Authorization: Bearer $TOKEN"  \
# MAGIC -H "Content-Type: application/json" \
# MAGIC -d '{"registered_model_name":"MLflow_Sklearn_Train_Tutorial"}' \
# MAGIC $ENABLE_URI

# COMMAND ----------

# MAGIC %md ### Start server - Form

# COMMAND ----------

# MAGIC %sh
# MAGIC ENABLE_URI="https://$HOST_NAME/2.0/preview/mlflow/endpoints/enable"
# MAGIC echo $ENABLE_URI
# MAGIC echo $MODEL_NAME
# MAGIC 
# MAGIC curl -v -H "Authorization: Bearer $TOKEN" --form registered_model_name=$MODEL_NAME \
# MAGIC -H "Content-Type: application/x-www-form-urlencoded" \
# MAGIC $ENABLE_URI

# COMMAND ----------

# MAGIC %md ### Run prediction with curl

# COMMAND ----------

# MAGIC %sh echo $SCORING_URI

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -s $SCORING_URI  \
# MAGIC -H "Authorization: Bearer $TOKEN" \
# MAGIC -H 'Content-Type: application/json' \
# MAGIC -d '{ 
# MAGIC   "columns": 
# MAGIC     [ "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol" ],
# MAGIC   "data": [
# MAGIC     [ 7,   0.27, 0.36, 20.7, 0.045, 45, 170, 1.001,  3,    0.45,  8.8 ], 
# MAGIC     [ 6.3, 0.3,  0.34,  1.6, 0.049, 14, 132, 0.994,  3.3,  0.49,  9.5 ], 
# MAGIC     [ 8.1, 0.28, 0.4,   6.9, 0.05,  30,  97, 0.9951, 3.26, 0.44, 10.1 ] 
# MAGIC   ] }'