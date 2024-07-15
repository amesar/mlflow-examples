# Databricks notebook source
# MAGIC %md ### Nested Runs Example
# MAGIC
# MAGIC ##### Widgets
# MAGIC * `1. Max levels` - number of nested levels - includes root level.
# MAGIC   * Max level of 2 will create a root run and a child run.
# MAGIC * `2. Max children` - number of runs per level.
# MAGIC * `3. Delete runs` - delete experiment runs before creating nested run.
# MAGIC
# MAGIC ##### Github
# MAGIC * https://github.com/amesar/mlflow-examples/tree/master/python/nested_runs

# COMMAND ----------

# MAGIC %md #### Setup

# COMMAND ----------

dbutils.widgets.text("1. Max levels",  "2")
max_levels = dbutils.widgets.get("1. Max levels")
max_levels = int(max_levels)

dbutils.widgets.text("2. Max children", "2")
max_children = dbutils.widgets.get("2. Max children")
max_children = int(max_children)

dbutils.widgets.dropdown("3. Delete runs", "yes", ["yes","no"]) 
delete_runs = dbutils.widgets.get("3. Delete runs") == "yes"

print("max_levels:", max_levels)
print("max_children:", max_children)
print("delete_runs:", delete_runs)

# COMMAND ----------

import mlflow

notebook = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(notebook)
experiment

# COMMAND ----------

# MAGIC %md #### Delete existing runs

# COMMAND ----------

if experiment and delete_runs:
    runs = client.search_runs(experiment.experiment_id)
    print(f"Deleting {len(runs)} runs")
    for run in runs:
        client.delete_run(run.info.run_id)

# COMMAND ----------

# MAGIC %md #### Create nested runs

# COMMAND ----------

def train(max_levels, max_children, level=0, child_idx=0):
    _TAB = "  "
    def _mk_tab(level):
        return  "".join([ _TAB for _ in range(level)])

    if level >= max_levels:
        return
    tab = _mk_tab(level)
    tab2 = tab + _TAB
    run_name = f"L_{level:02d}"
    run_name = f"L_{level:02d}_{child_idx:02d}"
    print(f"{tab}Level={level} Child={child_idx}")
    print(f"{tab2}run_name: {run_name}")
    with mlflow.start_run(run_name=run_name, nested=(level > 0)) as run:
        print(f"{tab2}run_id: {run.info.run_id}")
        mlflow.log_param("max_levels", max_levels)
        mlflow.log_param("max_children", max_children)
        mlflow.log_metric("auroch", 0.123)
        mlflow.set_tag("_run_name", run_name)
        with open("info.txt", "w", encoding="utf-8") as f:
            f.write(run_name)
        mlflow.log_artifact("info.txt")
        for j in range(max_children):
            train(max_levels, max_children, level+1, j)
    return run

# COMMAND ----------

run = train(max_levels, max_children)
run = client.get_run(run.info.run_id)

# COMMAND ----------

# MAGIC %md #### Display root run

# COMMAND ----------

run

# COMMAND ----------

run.info.run_id, run.info.run_name, run.data.tags.get("mlflow.rootRunId")

# COMMAND ----------

def show_runs(runs):
    runs = sorted(runs, key=lambda run: run.info.run_name)
    for run in runs:
        print(run.info.run_name, run.info.run_id, run.data.tags.get("mlflow.rootRunId"), run.data.tags.get("mlflow.parentRunId") )
    print(f"Number of runs: {len(runs)}")

# COMMAND ----------

# MAGIC %md #### All runs

# COMMAND ----------

runs = client.search_runs(run.info.experiment_id)
show_runs(runs)

# COMMAND ----------

# MAGIC %md #### Runs with a `mlflow.rootRunId` tag that equals the root run ID

# COMMAND ----------

filter =  f"tags.mlflow.rootRunId = '{run.info.run_id}'"
runs = client.search_runs(run.info.experiment_id, filter_string=filter)
show_runs(runs), filter

# COMMAND ----------

# MAGIC %md #### Runs with a `mlflow.parentRunId` tag

# COMMAND ----------

filter =  f"tags.mlflow.parentRunId like '%'"
runs = client.search_runs(run.info.experiment_id, filter_string=filter)
show_runs(runs)
