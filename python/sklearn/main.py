import click
from wine_quality.train import Trainer

@click.command()
@click.option("--experiment_name", help="Experiment name", default=None, type=str)
@click.option("--data_path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--model_name", help="Registered model name", default=None, type=str)
@click.option("--max_depth", help="Max depth", default=None, type=int)
@click.option("--max_leaf_nodes", help="Max leaf nodes", default=32, type=int)
@click.option("--output_path", help="Output file containing run ID", default="none", type=str)
@click.option("--log_as_onnx", help="Log model as ONNX flavor", default=False, type=bool)
@click.option("--run_origin", help="Run origin", default="none", type=str)

def main(experiment_name, data_path, model_name, max_depth, max_leaf_nodes, log_as_onnx, output_path, run_origin):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    model_name = None if not model_name or model_name == "None" else model_name
    print("Processed Options:")
    print(f"  model_name: {model_name} - type: {type(model_name)}")
    trainer = Trainer(experiment_name, data_path, log_as_onnx, run_origin)
    trainer.train(max_depth, max_leaf_nodes, model_name, output_path)

if __name__ == "__main__":
    main()
