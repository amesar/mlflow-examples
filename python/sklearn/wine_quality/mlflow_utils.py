import mlflow
from mlflow.exceptions import RestException

client = mlflow.client.MlflowClient()

def register_model(
        run,
        mlflow_model_name,
        registered_model_name,
        registered_model_version_stage,
        archive_existing_versions
    ):
    try:
        desc = "Skearn Wine Quality model"
        tags = { "info": desc}
        client.create_registered_model(registered_model_name, tags, desc)
    except RestException:
        pass
    source = f"{run.info.artifact_uri}/{mlflow_model_name}"
    print("MLflow model source:", source)
    vr = client.create_model_version(registered_model_name, source, run.info.run_id)
    if registered_model_version_stage:
        if  registered_model_version_stage in [ "None", "Archived" ]:
            print(f"WARNING: Cannot explicitly transition model '{registered_model_name}/{vr.version}' to stage '{registered_model_version_stage}'")
        else:
            print(f"Transitioning '{registered_model_name}/{vr.version}' to stage '{registered_model_version_stage}'")
            client.transition_model_version_stage(registered_model_name, vr.version, registered_model_version_stage, archive_existing_versions)
    desc = f"v{vr.version} {registered_model_version_stage} - wine"
    client.update_model_version(registered_model_name, vr.version, desc)
    client.set_model_version_tag(registered_model_name, vr.version, "registered_version_info", desc)
