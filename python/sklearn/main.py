from argparse import ArgumentParser
from wine_quality.train import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", required=False, type=str)
    parser.add_argument("--model_name", dest="model_name", help="Registered model name", default=None)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=None, type=int)
    parser.add_argument("--max_leaf_nodes", dest="max_leaf_nodes", help="max_leaf_nodes", default=32, type=int)
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="none")
    parser.add_argument("--log_as_onnx", dest="log_as_onnx", help="Log model as ONNX flavor", default=False, type=bool)
    parser.add_argument("--output_path", dest="output_path", help="Output path containing run ID", default=None)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    trainer = Trainer(args.experiment_name, args.data_path, args.log_as_onnx, args.run_origin)
    model_name = None if args.model_name is None or args.model_name == "None" else args.model_name
    trainer.train(args.max_depth, args.max_leaf_nodes, model_name, args.output_path)
