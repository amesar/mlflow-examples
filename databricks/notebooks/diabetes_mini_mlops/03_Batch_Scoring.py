# Databricks notebook source
# MAGIC %md ## Batch Scoring
# MAGIC * Scores the best model run from the [01_Train_Model]($01_Train_Model) notebook.
# MAGIC * Scores with native Sklearn, Pyfunc and UDF flavors.
# MAGIC * Sklearn and Pyfunc scoring is executed only on the driver node, whereas UDF scoring uses all nodes of the cluster.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("1. Registered model", "")
model_name = dbutils.widgets.get("1. Registered model")
assert_widget(model_name, "1. Registered model")

print("model_name:", model_name)

# COMMAND ----------

# MAGIC %md ### Prepare scoring data
# MAGIC * Drop the label column

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame 
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)
display(data)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
data_to_predict = test.drop(["progression"], axis=1)

# COMMAND ----------

# MAGIC
# MAGIC %md ### Score with native Sklearn flavor
# MAGIC * Executes only on the driver node of the cluster

# COMMAND ----------

import pandas as pd
model_uri = f"models:/{model_name}/1"
model_uri

# COMMAND ----------

model = mlflow.sklearn.load_model(model_uri)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions, columns=[_col_prediction]))

# COMMAND ----------

# MAGIC %md ### Score with Pyfunc flavor
# MAGIC * Executes only on the driver node of the cluster

# COMMAND ----------

import mlflow.pyfunc

model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(data_to_predict)
display(pd.DataFrame(predictions, columns=[_col_prediction]))

# COMMAND ----------

# MAGIC %md ### Distributed scoring with UDF
# MAGIC * Executes on all worker nodes of the cluster.
# MAGIC * UDF wraps the Sklearn model.
# MAGIC * Pass a Spark dataframe to the UDF.
# MAGIC * The dataframe is split into multiple pieces and sent to each worker in the cluster for scoring.

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn(_col_prediction, udf(*df.columns))
display(predictions.select(_col_prediction))
