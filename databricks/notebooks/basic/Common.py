# Databricks notebook source
# MAGIC %run ./Versions

# COMMAND ----------

import mlflow
client = mlflow.MlflowClient()

# COMMAND ----------

from mlflow.utils import databricks_utils
_host_name = databricks_utils.get_browser_hostname()
print("_host_name:", _host_name)

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    if _host_name:
        uri = f"https://{_host_name}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
        displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_uri(model_name):
    if _host_name:
        if "." in model_name: # is unity catalog model
            model_name = model_name.replace(".","/")
            uri = f"https://{_host_name}/explore/data/models/{model_name}"
        else:
            uri = f"https://{_host_name}/#mlflow/models/{model_name}"
        displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

# Test
#display_registered_model_uri("Sklearn_Wine")
#display_registered_model_uri("andre_catalog.ml_models.Sklearn_Wine_ws")

# COMMAND ----------

def display_registered_model_version_uri(model_name, version):
    if _host_name:
        if "." in model_name: # is unity catalog model
            model_name = model_name.replace(".","/")
            uri = f"https://{_host_name}/explore/data/models/{model_name}/version/{version}"
        else:
            uri = f"https://{_host_name}/#mlflow/models/{model_name}/versions/{version}"
        displayHTML("""<b>Registered Model Version URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_id_info(experiment_id):
    if _host_name:
        experiment = client.get_experiment(experiment_id)
        _display_experiment_info(experiment)

def _display_experiment_info(experiment):
    _host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
    uri = f"https://{_host_name}/#mlflow/experiments/{experiment.experiment_id}"
    displayHTML(f"""
    <table cellpadding=5 cellspacing=0 border=1 bgcolor="#FDFEFE" style="font-size:13px;">
    <tr><td colspan=2><b><i>Experiment</i></b></td></tr>
    <tr><td>UI link</td><td><a href="{uri}">{uri}</a></td></tr>
    <tr><td>Name</td><td>{experiment.name}</td></tr>
    <tr><td>ID</td><td>{experiment.experiment_id}</td></tr>
    </table>
    """)

# COMMAND ----------

def dump_obj(obj, title=None):
    title = title or type(obj).__name__
    print(f"{title}:")
    if obj:
        for k,v in obj.__dict__.items():
            print(f"  {k}: {v}")

def dump_obj_as_json(obj):
    import json
    print(json.dumps(obj.__dict__, indent=2))

# COMMAND ----------

def to_int(x):
  return None if x is None or x=="" else int(x)

# COMMAND ----------

def to_list_int(str, delimiter=" "): 
    return [ int(x) for x in str.split(delimiter) ]

# COMMAND ----------

import time
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
now = now()

# COMMAND ----------

def mk_dbfs_path(path):
    return path.replace("/dbfs","dbfs:") if path else None

def mk_local_path(path):
    return path.replace("dbfs:","/dbfs") if path else None

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0:
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

_model_version_stages = ["Production","Staging","Archived","None"]

# COMMAND ----------

from mlflow.exceptions import RestException

def delete_registered_model(model_name):
    """ Delete a model and all its versions """
    try:
        versions = client.get_latest_versions(model_name)
        print(f"Deleting {len(versions)} versions for model '{model_name}'")
        for v in versions:
            print(f"  version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
            client.transition_model_version_stage (model_name, v.version, "Archived") # 1.9.0
            client.delete_model_version(model_name, v.version)
        client.delete_registered_model(model_name)
    except RestException:
        pass

def register_model(run, 
        model_name, 
        model_version_stage = None, 
        archive_existing_versions = False, 
        model_alias = None,
        model_artifact = "model"
    ):
    """ Register mode with specified stage and alias """
    try:
       model =  client.create_registered_model(model_name)
    except RestException as e:
       model =  client.get_registered_model(model_name)
    source = f"{run.info.artifact_uri}/{model_artifact}"
    vr = client.create_model_version(model_name, source, run.info.run_id)
    if model_version_stage and model_version_stage != "None":
        print(f"Transitioning model '{model_name}/{vr.version}' to stage '{model_version_stage}'")
        client.transition_model_version_stage(model_name, vr.version, model_version_stage, archive_existing_versions=False)
    if model_alias:
        print(f"Setting model '{model_name}/{vr.version}' alias to '{model_alias}'")
        client.set_registered_model_alias(model_name, model_alias, vr.version)
    return vr

# COMMAND ----------

def register_model_uc(run, 
        reg_model_name, 
        reg_model_alias = None,
        run_model_artifact = "model"
    ):
    """ Register mode with specified alias """
    try:
       model =  client.create_registered_model(reg_model_name)
    except RestException as e:
       model =  client.get_registered_model(reg_model_name)
    source = f"{run.info.artifact_uri}/{run_model_artifact}"
    vr = client.create_model_version(reg_model_name, source, run.info.run_id)
    if reg_model_alias:
        print(f"Setting model '{reg_model_name}/{vr.version}' alias to '{reg_model_alias}'")
        client.set_registered_model_alias(reg_model_name, reg_model_alias, vr.version)
    return vr

# COMMAND ----------

import pandas as pd

class WineQuality():
    colLabel = "quality"
    colPrediction = "prediction"
    colFeatures = "features"
    _wine_color = "white"
    data_path = f"dbfs:/databricks-datasets/wine-quality/winequality-{_wine_color}.csv"

    @staticmethod
    def load_pandas_data():
        data_path = mk_local_path(WineQuality.data_path)
        print(f"Reading data as Pandas dataframe from '{data_path}'")
        df = pd.read_csv(data_path, delimiter=";")
        df.columns = df.columns.str.replace(" ","_")
        return df
    
    @staticmethod
    def _load_spark_data():
        print(f"Reading data as Spark dataframe from '{WineQuality.data_path}'")
        df = (spark.read.format("csv")
            .option("header", True)
            .option("inferSchema", True)
            .option("delimiter",";")
            .load(WineQuality.data_path) )
        columns = [ col.replace(" ","_") for col in df.columns ]
        df = df.toDF(*columns)
        return df
    
    @staticmethod
    def get_data(table_name=""):
        path = WineQuality.data_path
        if table_name == "":
            df = WineQuality._load_spark_data()
            return df, path
        else:
            if not spark.catalog._jcatalog.tableExists(table_name):
                print(f"Creating table '{table_name}'")
                df = WineQuality._load_spark_data()
                df.write.mode("overwrite").saveAsTable(table_name)
            df = spark.table(table_name)
            print(f"Reading table '{table_name}'")
            return df, table_name

    @staticmethod
    def prep_training_data(data):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=0.30, random_state=42)
        X_train = train.drop([WineQuality.colLabel], axis=1)                 
        X_test = test.drop([WineQuality.colLabel], axis=1)
        y_train = train[WineQuality.colLabel]
        y_test = test[WineQuality.colLabel]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def prep_prediction_data(data):
        return data.drop(WineQuality.colLabel, axis=1)

# COMMAND ----------

def log_data_input(run, log_input, data_source, df):
    if log_input and hasattr(run, "inputs"):
        print(f"Logging input data_source '{data_source}'")
        if data_source.startswith("dbfs"):
            dataset_name = data_source.split("/")[-1]
            if isinstance(df, pandas.core.frame.DataFrame):
                dataset = mlflow.data.from_pandas(df, source=mk_local_path(data_source), name=dataset_name)
            else:
                dataset = mlflow.data.from_spark(df, path=data_source, name=dataset_name)
            print(f"Logging input with Spark - dataset: '{dataset}'")
        else:
            dataset_name = data_source.split(".")[-1]
            dataset = mlflow.data.load_delta(table_name=data_source, name=dataset_name)
            print(f"Logging input with Delta - dataset: '{dataset}'")
        mlflow.log_input(dataset, context="training")
    else:
        print("Skipped logging input")

# COMMAND ----------

def show_mlflow_uris(msg):
    print(f"{msg}:")
    print("  mlflow.get_tracking_uri:", mlflow.get_tracking_uri())
    print("  mlflow.get_registry_uri:", mlflow.get_registry_uri())
    print("  mlflowClient.tracking_uri:", client.tracking_uri)
    print("  mlflowClient.registry_uri:", client._registry_uri)

# COMMAND ----------

def activate_unity_catalog():
    mlflow.set_registry_uri("databricks-uc")
    show_mlflow_uris("After UC settings")
    client = mlflow.MlflowClient() 
    return client

# COMMAND ----------

def is_unity_catalog(model_name):
    return len(model_name.split(".")) == 3

def toggle_unity_catalog(model_name):
    registry_uri = "databricks-uc" if is_unity_catalog(model_name) else "databricks"
    print(f"Setting new registry_uri URI: {registry_uri}")
    mlflow.set_registry_uri(registry_uri)
    print(f"New registry_uri URI: {mlflow.get_registry_uri()}")
    client = mlflow.MlflowClient()
    print(f"New client.registry_uri URI: {client._registry_uri}")
    return client

# COMMAND ----------


