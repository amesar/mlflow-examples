# Databricks notebook source
# MAGIC %md ## Sklearn Iris MLflow model - Autolog
# MAGIC
# MAGIC Simple Sklearn model using autologging
# MAGIC
# MAGIC * dbfs:/home/andre.mesarovic@databricks.com/data/iris.csv
# MAGIC * andre_data.iris
# MAGIC
# MAGIC TODO: Sync up data reading with Sklearn_Iris.

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Data path", "") 
data_path = dbutils.widgets.get("1. Data path")
if data_path=="": data_path = None

dbutils.widgets.text("2. Max depth", "1") 
max_depth = to_int(dbutils.widgets.get("2. Max depth"))

print("data_path:", data_path)
print("max_depth:", max_depth)

# COMMAND ----------

# MAGIC %md ### Get data

# COMMAND ----------

from sklearn import datasets
from sklearn.model_selection import train_test_split

col_label = "species"

def prep_training_data(data):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=0.30, random_state=42)
    train_x = train.drop([col_label], axis=1)                 
    test_x = test.drop([col_label], axis=1)
    train_y = train[col_label]
    test_y = test[col_label]
    return train_x, test_x, train_y, test_y

def get_data(data_path=None):
    data_path = mk_local_path(data_path)
    if data_path and not data_path.startswith("/dbfs"):
        df = spark.table(mk_dbfs_path(data_path))
        print(f"Loading data from table '{data_path}'")
        pdf = df.toPandas()
        X_train, X_test, y_train, y_test = prep_training_data(pdf)
    elif not data_path:
        print("Loading default data")
        dataset = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3)
    else:
        print(f"Loading data from '{data_path}'")
        import pandas as pd
        df = pd.read_csv(data_path)
        train, test = train_test_split(df, test_size=0.30, random_state=42)
        X_train = train.drop([col_label], axis=1)
        X_test = test.drop([col_label], axis=1)
        y_train = train[[col_label]]
        y_test = test[[col_label]]
    return X_train, X_test, y_train, y_test

# COMMAND ----------

X_train, X_test, y_train, y_test = get_data(data_path)

# COMMAND ----------

# MAGIC %md ### Train model

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=max_depth)
model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md ### Display UI links

# COMMAND ----------

run = mlflow.last_active_run() 
client.set_tag(run.info.run_id,"data_path", data_path)
print("run_id:", run.info.run_id)

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

display_experiment_id_info(run.info.experiment_id)
