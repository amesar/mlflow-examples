import mlflow
import click

print("MLflow Version:", mlflow.__version__)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

_TAB = "  "

def _mk_tab(level):
    return  "".join([ _TAB for j in range(level) ])

def _train(base_name, max_level, max_children, level=0, child_idx=0):
    if level >= max_level:
        return
    tab = _mk_tab(level)
    tab2 = tab + _TAB
    name = f"L_{level}"
    print(f"{tab}Level={level} Child={child_idx}")
    print(f"{tab2}name: {name} max_level: {max_level}")
    with mlflow.start_run(run_name=name, nested=(level > 0)) as run:
        print(f"{tab2}run_id: {run.info.run_id}")
        print(f"{tab2}experiment_id: {run.info.experiment_id}")
        mlflow.log_param("max_level", max_level)
        mlflow.log_param("max_children", max_children)
        mlflow.log_param("alpha", str(child_idx+0.1))
        mlflow.log_metric("auroch", 0.123)
        mlflow.set_tag("algo", name)
        with open("info.txt", "w", encoding="utf-8") as f:
            f.write(name)
        mlflow.log_artifact("info.txt")
        for j in range(max_children):
            _train(base_name, max_level, max_children, level+1, j)


@click.command()
@click.option("--experiment", help="Experiment name.", type=str, required=False)
@click.option("--max-level", help="Number of nested levels.", type=int, default=1)
@click.option("--max-children", help="Number of runs per level.", type=int, default=1)
def main(experiment, max_level, max_children):
    """
    Create a nested run with specified number of levels.
    """
    print("Options:")
    for k,v in locals().items(): print(f"  {k}: {v}")
    if experiment:
        mlflow.set_experiment(experiment)
    _train("nst",max_level,max_children)

if __name__ == "__main__":
    main()
