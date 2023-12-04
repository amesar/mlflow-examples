# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn import datasets

# COMMAND ----------

def create_pandas_df():
    print(f"Loading data from sklearn diabetes dataset")
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    columns = diabetes.feature_names + ['progression']

    return pd.DataFrame(d, columns=columns)

# COMMAND ----------

def load_data(table_name):
    if table_name:
        if not spark.catalog.tableExists(table_name):
            print(f"Creating table '{table_name}'")
            df = spark.createDataFrame(create_pandas_df())
            df.write.mode("overwrite").saveAsTable(table_name)
        else:
            df = spark.table(table_name)
            print(f"Loading data from table '{table_name}'")
        return df.toPandas() 
    else:
        return create_pandas_df()
