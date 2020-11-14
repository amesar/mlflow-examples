import pandas as pd
import h2o
from h2o.automl import H2OAutoML, get_leaderboard
import mlflow
import mlflow.h2o

print("MLflow version:", mlflow.__version__)
print("h2o version:", h2o.__version__)
h2o.init()
default_data_path = "../../data/train/wine-quality-white.csv"

def qname(clz):
    return str(clz).replace("<class '","").replace("'>","")

def prepare_data(data_path):
    data = h2o.import_file(path=data_path)
    r = data['quality'].runif()
    train_data = data[r < 0.7]
    test_data = data[0.3 <= r]
    train_cols = [n for n in data.col_names if n != "quality"]
    return train_data, test_data, train_cols

def train(data_path, max_models, model_name):
    train_data, test_data, train_cols = prepare_data(args.data_path)
    test_cols = train_cols[:-1]
    test_cols = "quality"

    with mlflow.start_run() as run:
        print("run_id:", run.info.run_id)
        model = H2OAutoML(max_models=max_models, max_runtime_secs=300, seed=24, nfolds=6)
        model.train(x=train_cols, y=test_cols, training_frame=train_data, validation_frame=test_data)
        mlflow.log_param("max_models", max_models)
        mlflow.log_metric("rmse", model.leader.rmse())

        mlflow.set_tag("mlflow_version", mlflow.__version__)
        mlflow.set_tag("h2o_version", h2o.__version__)
        mlflow.set_tag("model.leader.class",qname(model.leader.__class__))
        mlflow.set_tag("model.leader.estimator_type",model.leader. _estimator_type)
        mlflow.set_tag("num_leaderboard_models",model.leaderboard.nrows)

        lb = get_leaderboard(model, extra_columns='ALL')
        print(lb)

        path = "leaderboard.csv"
        h2o.export_file(lb, path=path, force=True)
        mlflow.log_artifact(path)

        from tabulate import tabulate
        df = lb.as_data_frame()
        table = tabulate(df, headers="keys", tablefmt="psql", showindex=False)
        path = "leaderboard.txt"
        with open(path, "w") as f:
            f.write(table)
        mlflow.log_artifact(path)

        df = df[["model_id"]]
        with open("models.csv", "w") as f:
            df.to_csv(f, index=False, header=False)
        mlflow.log_artifact("models.csv")

        mlflow.h2o.log_model(model.leader, "h2o-model", registered_model_name=args.model_name)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", required=False, type=str)
    parser.add_argument("--model_name", dest="model_name", help="Registered model name", default=None)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=default_data_path)
    parser.add_argument("--max_models", dest="max_models", help="max_models", default=5, type=int)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    client = mlflow.tracking.MlflowClient()
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    train(args.data_path, args.max_models, args.model_name)
