from argparse import ArgumentParser
import pandas as pd
from tabulate import tabulate
import mlflow
from mlflow import projects

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
client = mlflow.tracking.MlflowClient()

def dump_dict(dct, msg):
    print(f"  {msg}")
    for k,v in dct.items():
      print(f"    {k}: {v}")

def dump_run(run, msg, verbose):
    print(msg)
    exp = client.get_experiment(run.info.experiment_id)
    if verbose:
        print("  Info:")
        print("    experiment_name:",exp.name)
        dump_dict(run.info.__dict__,"Info")
        dump_dict(run.data.params,"Params")
        dump_dict(run.data.metrics,"Metrics")
        dump_dict(run.data.tags,"Tags")
    else:
        print("  run_id:",run.info.run_id)
        print("  experiment_name:",exp.name)
        print("  experiment_id:",exp.experiment_id)

def runs_equal(run1, run2):
    return \
        run1.data.metrics == run2.data.metrics 
        #and run1.data.params == run2.data.params

def run(run_id, uri, experiment_name, verbose):
    run1 = client.get_run(run_id)
    dump_run(run1,"Run1", verbose)

    version = run1.data.tags.get("mlflow.source.git.commit",None)
    if version is None:
        raise Exception(f"Missing tag 'mlflow.source.git.commit' for run {run_id}")

    # Run the run
    res = projects.run(uri, parameters=run1.data.params, version=version, experiment_name=experiment_name)
    print("Result:")
    print("  run_id:",res.run_id)
    print("  get_status:",res.get_status())

    # Print results of reproduced run
    run2 = client.get_run(res.run_id)
    dump_run(run2, "Run2", verbose)

    # Print comparison of metrics
    data = [ [k,v, run2.data.metrics[k]] for k,v in run1.data.metrics.items() ]
    df = pd.DataFrame(data, columns = ["Metric","Run1", "Run2"])
    print()
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    eq = runs_equal(run1,run2)
    print("Runs equal:",eq)

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
    run(args.run_id, args.uri, args.experiment_name, args.verbose)
