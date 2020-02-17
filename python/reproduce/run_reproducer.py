import pandas as pd
from tabulate import tabulate
import mlflow
from mlflow import projects
from mlflow.utils import mlflow_tags

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

#def runs_equal(run1, run2):
    #return run1.data.metrics == run2.data.metrics 

def runs_equal(run1, run2, rel_tol):
    import math
    for k1,v1 in run1.data.metrics.items():
        v2 = run2.data.metrics[k1]
        if not math.isclose(v1, v2, rel_tol=rel_tol):
            return False
    return True

def get_tag(run, tag_name):
    tag_val = run.data.tags.get(tag_name)
    if tag_val is None:
        raise Exception(f"Missing tag '{tag_name}' for run {run.info.run_id}")
    return tag_val

def reproduce_run(run_id, experiment_name, rel_tol=1e-09, verbose=False):
    # Get target run
    run1 = client.get_run(run_id)
    dump_run(run1,"Target Run", verbose)
    uri = get_tag(run1, mlflow_tags.MLFLOW_SOURCE_NAME)
    print(f"git_uri: {mlflow_tags.MLFLOW_SOURCE_NAME}: {uri}")
    version = get_tag(run1, mlflow_tags.MLFLOW_GIT_COMMIT)
    print("version:",version)

    # Execute the run - reproduced run
    res = projects.run(uri, parameters=run1.data.params, version=version, experiment_name=experiment_name)
    print("Reproduced Run Result:")
    print("  run_id:",res.run_id)
    print("  get_status:",res.get_status())

    # Print results of reproduced run
    run2 = client.get_run(res.run_id)
    dump_run(run2, "Reproduced Run", verbose)

    # Print metrics comparison between target and reproduced run
    data = [ [k,v, run2.data.metrics[k]] for k,v in run1.data.metrics.items() ]
    df = pd.DataFrame(data, columns = ["Metric","Run1", "Run2"])
    print()
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    eq = runs_equal(run1, run2, rel_tol)
    print("Runs equal:",eq)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", required=False, default="Default")
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--verbose", dest="verbose", help="Verbose", default=False, action='store_true')
    parser.add_argument("--rel_tol", dest="rel_tol", help="Relative tolerance", default=1e-09)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    reproduce_run(args.run_id, args.experiment_name, args.rel_tol, args.verbose)
