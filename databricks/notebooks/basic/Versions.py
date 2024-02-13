# Databricks notebook source
# MAGIC %md ##### Versions

# COMMAND ----------

def get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

def add_version(versions, pkg_name):
    try:
        import importlib
        pkg = importlib.import_module(pkg_name)
        versions.append([pkg_name, pkg.__version__])
    except ModuleNotFoundError as e:
        print(f"{pkg_name}: not installed")

# COMMAND ----------

import os, sys, platform

versions = [
    [ "$DATABRICKS_RUNTIME_VERSION:", os.environ.get("DATABRICKS_RUNTIME_VERSION") ],
    [ "spark.version", spark.version ]
]
version_names = [
    "mlflow",
    "pyspark",
    "sklearn",
    "tensorflow",
    "keras",
    "torch",
    "xgboost",
    "onnx",
    "cloudpickle",
    "protobuf",
    "pandas",
    "numpy"
]
for vr in version_names:
    add_version(versions, vr)

versions.append(["$SCALA_VERSION:", os.environ.get("SCALA_VERSION")])
versions.append(["python version:", sys.version.replace("\n"," ")])
versions.append(["platform:", platform.platform()])

# COMMAND ----------

import pandas as pd
from tabulate import tabulate
df = pd.DataFrame(versions, columns = ["Name","Version"])
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

# COMMAND ----------

#%scala
#import org.mlflow.tracking.MlflowClientVersion
#println("Scala version: "+scala.util.Properties.versionString)
#println("Scala MLflow version: "+MlflowClientVersion.getClientVersion())
#spark.conf.set("mlflow.version.scala", MlflowClientVersion.getClientVersion())

# COMMAND ----------

# Warning if Python and Scala mlflow versions do not match
#mlflow_version_scala = spark.conf.get("mlflow.version.scala")
#if mlflow.__version__ != mlflow_version_scala:
#    print(f"WARNING: MLflow versions do not match: Python {mlflow.__version__} and Scala {mlflow_version_scala}")

# COMMAND ----------

# MAGIC %sh java -version
