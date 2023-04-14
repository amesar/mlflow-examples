# Databricks notebook source
# MAGIC %md # Basic XGBoost MLflow train and predict 
# MAGIC * Trains and saves model as XGBoost model.
# MAGIC * Predicts using XGBoost and pyfunc UDF flavors.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

# Default values per: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

dbutils.widgets.text("Estimators", "100") 
dbutils.widgets.text("Max Depth", "3") 
dbutils.widgets.text("Min Child Weight", "1.5")
max_depth = int(dbutils.widgets.get("Max Depth"))

estimators = int(dbutils.widgets.get("Estimators"))
max_depth = int(dbutils.widgets.get("Max Depth"))
min_child_weight = float(dbutils.widgets.get("Min Child Weight"))
estimators, max_depth, min_child_weight

# COMMAND ----------

import xgboost as xgb
print("MLflow Version:", mlflow.__version__)
print("XGBoost version:", xgb.__version__)

# COMMAND ----------

# MAGIC %md ### Prepare data

# COMMAND ----------

data = WineQuality.get_data()
train_x, test_x, train_y, test_y = WineQuality.prep_training_data(data)
display(data)

# COMMAND ----------

data.describe()

# COMMAND ----------

# MAGIC %md ### Train

# COMMAND ----------

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# COMMAND ----------

with mlflow.start_run() as run:
    run_id = run.info.run_uuid
    print("MLflow:")
    print("  run_id:",run_id)
    print("  experiment_id:",run.info.experiment_id)
    print("Parameters:")
    print("  estimators:",estimators)
    print("  max_depth:",max_depth)
    print("  min_child_weight:",min_child_weight)
    mlflow.log_param("estimators", estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_child_weight", min_child_weight)
    
    # MLflow tags
    mlflow.set_tag("mlflow_version", mlflow.__version__)
    mlflow.set_tag("xgboost_version", xgb.__version__)
    mlflow.set_tag("spark_version", spark.version)
    mlflow.set_tag("dbr_version", os.environ.get('DATABRICKS_RUNTIME_VERSION',None))
    
    model = xgb.XGBRegressor(
        n_estimators=estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        random_state=42)
    model = xgb.XGBRegressor()
    print(model)
    model.fit(train_x, train_y)
    mlflow.xgboost.log_model(model, "model")

    predictions = model.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, predictions))
    r2 = r2_score(test_y, predictions)
    print("Metrics:")
    print("  rmse:",rmse)
    print("  r2:",r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 

# COMMAND ----------

model.get_xgb_params()

# COMMAND ----------

display_run_uri(run.info.experiment_id, run_id)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

model_uri = f"runs:/{run_id}/model"
data_to_predict = data.drop(WineQuality.colLabel, axis=1)
labels = data[WineQuality.colLabel]
model_uri

# COMMAND ----------

# MAGIC %md #### Predict as xgboost

# COMMAND ----------

# Started failing with new XGBoost releases: Unknown data type: <class 'xgboost.core.DMatrix'>, trying to convert it to csr_matrix

try:
    model = mlflow.xgboost.load_model(model_uri)
    dtrain = xgb.DMatrix(data_to_predict, label=labels)
    predictions = model.predict(dtrain)
    type(predictions), predictions.shape
except Exception as e:
    print("ERROR:", e)

# COMMAND ----------

display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as pyfunc

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
data_to_predict = WineQuality.prep_prediction_data(data)
type(predictions), predictions.shape

# COMMAND ----------

display(pd.DataFrame(predictions, columns=[WineQuality.colPrediction]))

# COMMAND ----------

# MAGIC %md #### Predict as UDF

# COMMAND ----------

df = spark.createDataFrame(data_to_predict)
udf = mlflow.pyfunc.spark_udf(spark, model_uri)
predictions = df.withColumn(WineQuality.colPrediction, udf(*df.columns))
display(predictions)

# COMMAND ----------

type(predictions)
