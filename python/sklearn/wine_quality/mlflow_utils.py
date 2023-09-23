import numpy as np
import json
import mlflow
from mlflow.exceptions import RestException

client = mlflow.client.MlflowClient()


def register_model(
        run,
        mlflow_model_name,
        registered_model_name,
        registered_model_version_stage,
        archive_existing_versions,
        registered_model_alias
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
            print(f"Transitioning model '{registered_model_name}/{vr.version}' to stage '{registered_model_version_stage}'")
            client.transition_model_version_stage(registered_model_name, vr.version, registered_model_version_stage, archive_existing_versions)
    if registered_model_alias:
        print(f"Setting model '{registered_model_name}/{vr.version}' alias to '{registered_model_alias}'")
        client.set_registered_model_alias(registered_model_name, registered_model_alias, vr.version)
        client.set_model_version_tag(registered_model_name, vr.version, "alias", registered_model_alias)
    desc = f"v{vr.version} {registered_model_version_stage} - wine"
    client.update_model_version(registered_model_name, vr.version, desc)
    client.set_model_version_tag(registered_model_name, vr.version, "registered_version_info", desc)


def log_dict(dct, artifact_name):
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, artifact_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(dct, cls=NumpyEncoder))
        mlflow.log_artifact(path)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)
