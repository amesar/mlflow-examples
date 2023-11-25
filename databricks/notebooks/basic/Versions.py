# Databricks notebook source
# MAGIC %md ##### Versions

# COMMAND ----------

def get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

import os, sys, pyspark, sklearn, tensorflow, xgboost, torch, mlflow, platform, cloudpickle
import numpy
import pandas
try:
    import keras
    is_keras2 = False
except ModuleNotFoundError as e:
    import tensorflow.keras as keras
    is_keras2 = True
    
#print("python version:", sys.version.replace("\n"," "))
#print("DATABRICKS_RUNTIME_VERSION:", os.environ.get("DATABRICKS_RUNTIME_VERSION",None))
#print("SPARK_SCALA_VERSION:", os.environ.get("SPARK_SCALA_VERSION",None))
#print("SCALA_VERSION:", os.environ.get("SCALA_VERSION",None))
#print("sparkVersion:", get_notebook_tag("sparkVersion"))
#print("spark.version:", spark.version)
#print("pyspark.version:", pyspark.__version__)
#print("python mlflow version:", mlflow.__version__)
#print("mleap version", mleap.version.__version__)
#print("sklearn version:", sklearn.__version__)
#print("tensorflow version:", tensorflow.__version__)
#print("keras version:", keras.__version__)
#print("torch version:", torch.__version__)
#print("xgboost version:", xgboost.__version__)

# COMMAND ----------

lst = []
lst.append(["mlflow version:", mlflow.__version__])
lst.append(["$DATABRICKS_RUNTIME_VERSION:", os.environ.get("DATABRICKS_RUNTIME_VERSION",None)])
lst.append(["sparkVersion:", get_notebook_tag("sparkVersion")])
lst.append(["spark.version:", spark.version])
lst.append(["pyspark.version:", pyspark.__version__])
lst.append(["sklearn version:", sklearn.__version__])
lst.append(["tensorflow version:", tensorflow.__version__])
lst.append(["keras version:", keras.__version__])
lst.append(["torch version:", torch.__version__])
lst.append(["xgboost version:", xgboost.__version__])
lst.append(["cloudpickle version:", cloudpickle.__version__])
lst.append(["cloudpickle DEFAULT_PROTOCOL:", cloudpickle.DEFAULT_PROTOCOL])
lst.append(["pandas version:", pandas.__version__])
lst.append(["numpy version:", numpy.__version__])

try:
    from google import protobuf 
    lst.append(["protobuf version:", protobuf.__version__])
except AttributeError as e:
    lst.append(["protobuf version", None])

lst.append(["$SCALA_VERSION:", os.environ.get("SCALA_VERSION",None)])
lst.append(["python version:", sys.version.replace("\n"," ")])
lst.append(["platform:", platform.platform()])

# COMMAND ----------

try:
    import onnx
    lst.append(["onnx version:",onnx.__version__])
except ModuleNotFoundError as e:
    pass
    
try:
    import onnxmltools
    lst.append(["onnxmltools version:",onnxmltools.__version__])
except ModuleNotFoundError as e:
    pass
    
try:
    import onnxruntime
    lst.append(["onnxruntime version:",onnxruntime.__version__])
except ModuleNotFoundError as e:
    pass

try:
    import skl2onnx
    lst.append(["skl2onnx version:",skl2onnx.__version__])
except ModuleNotFoundError as e:
    pass

# COMMAND ----------

import pandas as pd
from tabulate import tabulate
df = pd.DataFrame(lst, columns = ["Name","Version"])
print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.mlflow.tracking.MlflowClientVersion
# MAGIC println("Scala version: "+scala.util.Properties.versionString)
# MAGIC println("Scala MLflow version: "+MlflowClientVersion.getClientVersion())
# MAGIC spark.conf.set("mlflow.version.scala", MlflowClientVersion.getClientVersion())

# COMMAND ----------

# Warning if Python and Scala mlflow versions do not match
mlflow_version_scala = spark.conf.get("mlflow.version.scala")
if mlflow.__version__ != mlflow_version_scala:
    print(f"WARNING: MLflow versions do not match: Python {mlflow.__version__} and Scala {mlflow_version_scala}")

# COMMAND ----------

# MAGIC %sh java -version
