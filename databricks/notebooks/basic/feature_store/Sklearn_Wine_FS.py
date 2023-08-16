# Databricks notebook source
# MAGIC %md ## Sklearn MLflow train and predict with feature store
# MAGIC
# MAGIC **Overview**
# MAGIC * Trains and predict using the feature store
# MAGIC * Run  notebook [Sklearn_Wine_FS]($Sklearn_Wine_FS) before running this notebook.
# MAGIC
# MAGIC **Widgets**
# MAGIC * `1. Experiment name` - if not set, use notebook experiment
# MAGIC * `2. Registered model` - if set, register as model
# MAGIC * `3. Feature table` - feature table
# MAGIC * `3. Max depth` 
# MAGIC * `5. Unity Catalog` 

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

dbutils.widgets.text("1. Experiment name", "")
dbutils.widgets.text("2. Registered model", "")
dbutils.widgets.text("3. Feature table", "")
dbutils.widgets.text("4. Max depth", "3")
dbutils.widgets.dropdown("5. Unity Catalog", "no", ["yes","no"])

experiment_name = dbutils.widgets.get("1. Experiment name")
model_name = dbutils.widgets.get("2. Registered model")
fs_table_name = dbutils.widgets.get("3. Feature table")
max_depth = int(dbutils.widgets.get("4. Max depth"))
use_uc = dbutils.widgets.get("5. Unity Catalog") == "yes"

fs_datapath = "/databricks-datasets/wine-quality/winequality-white.csv"

model_name = model_name or None
experiment_name = experiment_name or None

print("experiment_name:", experiment_name)
print("model_name:", model_name)
print("fs_table_name:", fs_table_name)
print("fs_datapath:", fs_datapath)
print("max_depth:", max_depth)
print("use_uc:", use_uc)

# COMMAND ----------

assert_widget(fs_table_name, "3. Feature table")

# COMMAND ----------

if experiment_name:
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print("Experiment:", exp.experiment_id, exp.name)

# COMMAND ----------

if use_uc:
    client = activate_unity_catalog()
    print("New client._registry_uri:",client._registry_uri)

# COMMAND ----------

# MAGIC %md ### Data

# COMMAND ----------

df = create_id_df(fs_datapath)

# COMMAND ----------

# inference_data_df includes wine_id (primary key), quality (prediction target), and a real time feature
from pyspark.sql.functions import rand

inference_data_df = df.select("wine_id", "quality", (10 * rand()).alias("real_time_measurement"))
display(inference_data_df)

# COMMAND ----------

from databricks.feature_store.client import FeatureStoreClient
fs_client = FeatureStoreClient()

# COMMAND ----------

from databricks.feature_store import FeatureLookup
from sklearn.model_selection import train_test_split

def load_data(table_name, lookup_key):
    # In the FeatureLookup, if you do not provide the `feature_names` parameter, all features except primary keys are returned
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set looks up features in model_feature_lookups that match the primary key from inference_data_df
    training_set = fs_client.create_training_set(
        inference_data_df, 
        model_feature_lookups, 
        label="quality", 
        exclude_columns="wine_id"
    )
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("quality", axis=1)
    y = training_pd["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, training_set

# COMMAND ----------

# Create the train and test datasets
X_train, X_test, y_train, y_test, training_set = load_data(fs_table_name, "wine_id")
X_train.head()

# COMMAND ----------

# MAGIC %md ### MLflow train

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(fs_client, X_train, X_test, y_train, y_test, training_set):
    with mlflow.start_run() as run:
        print("Run ID:", run.info.run_id)
        print("Run name:", run.info.run_name)
        model = RandomForestRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
 
        mlflow.log_metric("test_mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("test_r2_score", r2_score(y_test, y_pred))
        mlflow.set_tag("fs_table", fs_table_name)
        mlflow.set_tag("fs_data_path", fs_datapath)
        if model_name:
            mlflow.set_tag("registered_model", model_name)

        fs_client.log_model(
            model = model,
            artifact_path = "model",
            flavor = mlflow.sklearn,
            training_set = training_set
        )
        return run

# COMMAND ----------

run = train_model(fs_client, X_train, X_test, y_train, y_test, training_set)
run = client.get_run(run.info.run_id)
dump_obj(run.info, "Run Info")

# COMMAND ----------

# MAGIC %md ### Display UI links

# COMMAND ----------

display_run_uri(run.info.experiment_id, run.info.run_id)

# COMMAND ----------

display_experiment_id_info(run.info.experiment_id)

# COMMAND ----------

if model_name:
    version = register_model(run, model_name)
    display_registered_model_version_uri(model_name, version.version)

# COMMAND ----------

# MAGIC %md ### Predict

# COMMAND ----------

#model_uri = f"models:/{model_name}/latest"
model_uri = f"runs:/{run.info.run_id}/model"
model_uri

# COMMAND ----------

batch_input_df = inference_data_df.drop("quality") # Drop the label column
predictions_df = fs_client.score_batch(model_uri,  batch_input_df)                         
display(predictions_df)

# COMMAND ----------

display(predictions_df["wine_id", "prediction"])

# COMMAND ----------

# MAGIC %md ### Display Model Info

# COMMAND ----------

model_info = mlflow.models.get_model_info(model_uri)
#dump_obj_as_json(model_info) # fails if UC
dump_obj(model_info)

# COMMAND ----------

# MAGIC %md ### Pyfunc predict

# COMMAND ----------

# FAILS: ModuleNotFoundError: No module named 'databricks.feature_store.mlflow_model'
#from databricks.feature_store import mlflow_model

model = mlflow.pyfunc.load_model(model_uri)
model
