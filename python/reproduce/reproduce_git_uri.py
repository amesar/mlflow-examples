from argparse import ArgumentParser
import pandas as pd
from tabulate import tabulate
import mlflow
from mlflow import projects
import reproducer

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
client = mlflow.tracking.MlflowClient()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", required=False, default="Default")
    parser.add_argument("--uri", dest="uri", help="uri", required=True)
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--verbose", dest="verbose", help="Verbose", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    run = client.get_run(args.run_id)
    reproducer.dump_run(run,"Run1", args.verbose)
    reproducer.run(run, args.uri, args.experiment_name, args.verbose)
