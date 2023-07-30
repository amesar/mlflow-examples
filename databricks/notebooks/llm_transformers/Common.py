# Databricks notebook source
import mlflow
client = mlflow.MlflowClient()

# COMMAND ----------

def is_unity_catalog(name):
    return len(name.split(".")) == 3

# COMMAND ----------

def get_client(model_name):
    if is_unity_catalog(model_name):
        client = mlflow.MlflowClient(registry_uri="databricks-uc")
    else:
        client = mlflow.MlflowClient(registry_uri="databricks")
    print("mlflow.MlflowClient._registry_uri:", client._registry_uri)
    return client

# COMMAND ----------

def assert_widget(value, name):
    if len(value.rstrip())==0:
        raise Exception(f"ERROR: '{name}' widget is required")

# COMMAND ----------

def create_registered_model(client,  model_name):
    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model '{model_name}'")
    except mlflow.exceptions.RestException as e:
        print(f"Registered model '{model_name}' already exists")

# COMMAND ----------


def dict_as_json(dct, sort_keys=None):
    import json
    return json.dumps(dct, sort_keys=sort_keys, indent=2)

def dump_dict_as_json(dct):
    print(dict_as_json(dct))

def dump_obj(obj):
    title = type(obj).__name__
    print(f"{title}:")
    for k,v in obj.__dict__.items():
        print(f"  {k}: {v}")

def dump_flavor(model_info, flavor_name="transformers"):
    flavors = model_info.flavors
    flavor = flavors.get(flavor_name)
    dump_dict_as_json(flavor)

# COMMAND ----------

def create_transformer_tags(model_info):
    flavor = model_info.flavors.get("transformers")
    #return  { f"hf.{k}":v for k,v in flavor.items() } # UC: doesn't like dot "."
    return  { f"hf_{k}":v for k,v in flavor.items() }

# COMMAND ----------

def add_transformer_tags(client, model_info):
    """
    Add 'transformers' flavor as run tags for searchability.
    """
    tags = create_transformer_tags(model_info)
    for k,v in tags.items():
        client.set_tag(model_info.run_id, k, v)
    return tags
