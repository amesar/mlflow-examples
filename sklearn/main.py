from argparse import ArgumentParser
from wine_quality.train import Trainer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../data/wine-quality-white.csv")
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=None, type=int)
    parser.add_argument("--max_leaf_nodes", dest="max_leaf_nodes", help="max_leaf_nodes", default=None, type=int)
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="none")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    trainer = Trainer(args.experiment_name, args.data_path, args.run_origin)
    trainer.train(args.max_depth, args.max_leaf_nodes)
