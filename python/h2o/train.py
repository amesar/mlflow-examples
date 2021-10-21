import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

import mlflow
import mlflow.h2o

print("MLflow version:", mlflow.__version__)
print("h2o version:", h2o.__version__)
h2o.init()
default_data_path = "../../data/train/wine-quality-white.csv"

def prepare_data(data_path):
    data = h2o.import_file(path=data_path)
    r = data['quality'].runif()
    train_data = data[r < 0.7]
    test_data = data[0.3 <= r]
    train_cols = [n for n in data.col_names if n != "quality"]
    return train_data, test_data, train_cols

def train(data_path, ntrees, log_as_onnx, model_name):
    train_data, test_data, train_cols = prepare_data(args.data_path)
    with mlflow.start_run() as run:
        exp = client.get_experiment(run.info.experiment_id)
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        print("  experiment_name:", exp.name)
        print("  experiment_artifact_location:", exp.artifact_location)
        rf = H2ORandomForestEstimator(ntrees=ntrees)
        rf.train(train_cols, "quality", training_frame=train_data, validation_frame=test_data)

        mlflow.log_param("ntrees", ntrees)

        mlflow.log_metric("rmse", rf.rmse())
        mlflow.log_metric("r2", rf.r2())
        mlflow.log_metric("mae", rf.mae())

        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.set_tag("h2o_version", h2o.__version__)

        mlflow.h2o.log_model(rf, "h2o-model", registered_model_name=args.model_name)

        if log_as_onnx:
            import onnxmltools
            from onnxmltools.convert import convert_h2o
            print("onnxmltools.version:",onnxmltools.__version__)
            path = f"{exp.artifact_location}/{run.info.run_id}/artifacts/h2o-model/model.h2o"
            onnx_model = convert_h2o(path)
            print("onnx_model.type:",type(onnx_model))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment-name", dest="experiment_name", help="experiment_name", required=False, type=str)
    parser.add_argument("--model-name", dest="model_name", help="Registered model name", default=None)
    parser.add_argument("--data-path", dest="data_path", help="Data path", default=default_data_path)
    parser.add_argument("--ntrees", dest="ntrees", help="ntrees", default=5, type=int)
    parser.add_argument("--log-as-onnx", dest="log_as_onnx", help="Log model as ONNX flavor", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    client = mlflow.tracking.MlflowClient()
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    train(args.data_path, args.ntrees, args.log_as_onnx, args.model_name)
