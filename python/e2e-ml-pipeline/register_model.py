import time
import mlflow
import mlflow.pyfunc
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import common

client = mlflow.tracking.MlflowClient()
print(f"MLflow Version: {mlflow.__version__}")

def show_version(version):
    print(f"Version: id={version.version} status={version.status} state={version.current_stage}")

def fmt_version(version):
    return f"Version: id={version.version} status={version.status} state={version.current_stage}"

""" Create registered model if it doesn't exist and remove any existing versions """
def init(model_name):
    from mlflow.exceptions import RestException
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Found model '{model_name}'")
        versions = client.get_latest_versions(model_name)
        print(f"Found {len(versions)} versions for model '{model_name}'")
        for v in versions:
            print(f"  version={v.version} status={v.status} stage={v.current_stage} run_id={v.run_id}")
            client.transition_model_version_stage (model_name, v.version, "Archived")
            client.delete_model_version(model_name, v.version)
    except RestException as e:
        print(f"INFO: {e}")
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            print(f"Creating {model_name}")
            registered_model = client.create_registered_model(model_name)
        else:
            print(f"ERROR: {e}")

""" Due to blob eventual consistency, wait until a newly created version is READY state """
def wait_until_version_ready(model_name, model_version, sleep_time=1, iterations=100):
    start = time.time()
    for _ in range(iterations):
        version = client.get_model_version(model_name, model_version.version)
        status = ModelVersionStatus.from_string(version.status)
        show_version(version)
        if status == ModelVersionStatus.READY:
            break
        time.sleep(sleep_time)
    end = time.time()
    print(f"Waited {round(end-start,2)} seconds")

def run(experiment_name, data_path, model_name):
    print(f"==== {__file__} ====")

    # Get best run
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(exp.experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id} {best_run.data.metrics['rmse']}")

    init(model_name)
    registered_model = client.get_registered_model(model_name)

    # Create new model version
    source = f"{best_run.info.artifact_uri}/sklearn-model"
    version = client.create_model_version(model_name, source, best_run.info.run_id)

    # Wait unti version is in READY status
    wait_until_version_ready(model_name, version, sleep_time=2)
    version = client.get_model_version(model_name,version.version)
    version_id = version.version
    show_version(version)

    # Promote version to production stage
    client.transition_model_version_stage (model_name, version_id, "Production")
    version = client.get_model_version(model_name, version_id)
    show_version(version)

    # Get data to score
    _, X_test, _, _ = common.build_data(data_path)

    # Fetch production model and score data
    model_uri = f"models:/{model_name}/production"
    model = mlflow.pyfunc.load_model(model_uri)
    predictions = model.predict(X_test)
    print(f"predictions: {predictions}")
 
    return model_uri

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default=common.experiment_name)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=common.data_path)
    parser.add_argument("--model_name", dest="model_name", help="Model name", default=common.model_name)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    run(args.experiment_name, args.data_path, args.model_name)
