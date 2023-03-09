# Databricks notebook source
# /Repos/andre.mesarovic@databricks.com/mlflow-examples/databricks/notebooks/Mini_MLOps_Pipeline/scratch

# COMMAND ----------

def create_scratch_experiment():
    with mlflow.start_run() as run:
        mlflow.set_tag("info","hi there")
        print("run_id:", run.info.run_id)
        print("experiment_id:", run.info.experiment_id)

    client = mlflow.client.MlflowClient()
    exp = client.get_experiment(run.info.experiment_id)
    print()
    print("exp:",exp)
    return exp.name
    #print("experiment_id:", exp.experiment_id)
