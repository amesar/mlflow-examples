# Databricks notebook source
# MAGIC %run ./Versions

# COMMAND ----------

import mlflow
client = mlflow.client.MlflowClient()

# COMMAND ----------

from mlflow.utils import databricks_utils
host_name = databricks_utils.get_browser_hostname()
print("host_name:", host_name)

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    if host_name:
        uri = f"https://{host_name}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
        displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_uri(model_name):
    if host_name:
        uri = f"https://{host_name}/#mlflow/models/{model_name}"
        displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_version_uri(model_name, version):
    if host_name:
        uri = f"https://{host_name}/#mlflow/models/{model_name}/versions/{version}"
        displayHTML("""<b>Registered Model Version URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_id_info(experiment_id):
    if host_name:
        experiment = client.get_experiment(experiment_id)
        _display_experiment_info(experiment)

def _display_experiment_info(experiment):
    host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()
    uri = f"https://{host_name}/#mlflow/experiments/{experiment.experiment_id}"
    displayHTML(f"""
    <table cellpadding=5 cellspacing=0 border=1 bgcolor="#FDFEFE" style="font-size:13px;">
    <tr><td colspan=2><b><i>Experiment</i></b></td></tr>
    <tr><td>UI link</td><td><a href="{uri}">{uri}</a></td></tr>
    <tr><td>Name</td><td>{experiment.name}</td></tr>
    <tr><td>ID</td><td>{experiment.experiment_id}</td></tr>
    </table>
    """)

# COMMAND ----------

def to_int(x):
  return None if x is None or x=="" else int(x)

# COMMAND ----------

def to_list_int(str, delimiter=" "): 
    return [ int(x) for x in str.split(delimiter) ]

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
    if model_version_stage:
        print(f"Transitioning model '{model_name}/{vr.version}' to stage '{model_version_stage}'")
        client.transition_model_version_stage(model_name, vr.version, model_version_stage, archive_existing_versions=False)
    if model_alias:
        print(f"Setting model '{model_name}/{vr.version}' alias to '{model_alias}'")
        client.set_registered_model_alias(model_name, model_alias, vr.version)
    return vr

# COMMAND ----------

import time
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
now = now()

# COMMAND ----------

class WineQuality():
    colLabel = "quality"
    colPrediction = "prediction"
    colFeatures = "features"

    @staticmethod
    def get_data(table_name=""):
        """ Return data as Pandas dataframe """
        import pandas as pd
        path = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
        if table_name == "":
            print(f"Reading data from '{path}'")
            pdf = pd.read_csv(path)
            pdf.columns = pdf.columns.str.replace(" ","_") # for consistency with Spark column names
            return pdf
        else:
            if not spark.catalog._jcatalog.tableExists(table_name):
                print(f"Creating table '{table_name}'")
                pdf = pd.read_csv(path)
                pdf.columns = pdf.columns.str.replace(" ","_") # make Spark legal column names
                df = spark.createDataFrame(pdf)
                df.write.mode("overwrite").saveAsTable(table_name)
            print(f"Using table '{table_name}'")
            return spark.table(table_name).toPandas()

    @staticmethod
    def prep_training_data(data):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=0.30, random_state=42)
        train_x = train.drop([WineQuality.colLabel], axis=1)                 
        test_x = test.drop([WineQuality.colLabel], axis=1)
        train_y = train[WineQuality.colLabel]
        test_y = test[WineQuality.colLabel]
        return train_x, test_x, train_y, test_y

    @staticmethod
    def prep_prediction_data(data):
        return data.drop(WineQuality.colLabel, axis=1)

# COMMAND ----------

def mk_dbfs_path(path):
    return path.replace("/dbfs","dbfs:")

def mk_local_path(path):
    return path.replace("dbfs:","/dbfs")

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0:
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

_model_version_stages = ["Production","Staging","Archived","None"]
