# Databricks notebook source
# MAGIC %md # Real-time scoring model from Model Registry
# MAGIC * Real-time scoring against MLflow model server with curl.
# MAGIC * Scores the best model run created in the [01_Train_Model]($01_Train_Model) notebook and registered in the [03_Register_Model]($03_Register_Model) notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC First start the model serving cluster: 
# MAGIC   * Navigate to the https://demo.cloud.databricks.com/#mlflow/models/MLflow MLOps E2E Pipeline/serving cluster page.
# MAGIC   MLflow MLOps E2E Pipeline
# MAGIC   * Click the `Enable Serving` button and wait until cluster is in `Ready Status`.
# MAGIC   * Then proceeed to run this notebook.
# MAGIC   
# MAGIC <img src="https://github.com/amesar/mlflow-resources/blob/master/images/databricks/model_server/enable_serving.png?raw=true" width="700" />

# COMMAND ----------

# MAGIC %md ### Setup

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
os.environ["TOKEN"] = context.apiToken().get() # 
scoring_uri

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