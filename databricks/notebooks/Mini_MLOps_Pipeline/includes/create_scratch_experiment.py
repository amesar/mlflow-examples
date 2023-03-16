# Databricks notebook source
def create_scratch_experiment():
    print("Creating scratch experiment")
    with mlflow.start_run() as run:
        mlflow.set_tag("info","hi there")
        #print("run_id:", run.info.run_id)
        #print("experiment_id:", run.info.experiment_id)

    client = mlflow.client.MlflowClient()
    exp = client.get_experiment(run.info.experiment_id)
    #print()
    #print(">> Create_scratch: exp:",exp)
    #print(f">> Created scratch experiment '{exp.name}'")
    return exp
