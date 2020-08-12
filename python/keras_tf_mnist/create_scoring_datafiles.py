
"""
Generate files for scoring input to the MLflow scoring server and TensorFlow Serving server.
Each server expects a JSON file with its own schema.
Data is obtained from the standard Keras MNIST 'x_test' field from ~/.keras/datasets/mnist.npz.
"""

import os
import json
import utils

def write_json(data, opath):
    with open(opath, "w") as f:
        f.write(json.dumps(data,indent=2)+"\n")

def to_json_mlflow(data, opath):
    data = data.tolist()
    line = data[0]
    print("Number of columns:",len(line))
    columns = [ f"col_{j}" for j in range(0,len(line)) ]
    dct = { "columns": columns, "data": data }
    write_json(dct, opath)

def to_json_tensorflow_serving(data, opath):
    dct = { "instances": data.tolist() }
    write_json(dct, opath)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--rows", dest="rows", help="Number of rows", default=None, type=int)
    parser.add_argument("--base_name", dest="base_name", help="Base name", default="mnist", type=str)
    parser.add_argument("--output_dir", dest="output_dir", help="Output directory", default=".")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    x_test = utils.get_prediction_data()
    print("x_test.type:",type(x_test))
    print("x_test.shape:",x_test.shape)
    if args.rows:
        x_test = x_test[:args.rows]
    print("x_test.shape:",x_test.shape)

    to_json_mlflow(x_test, os.path.join(args.output_dir,f"{args.base_name}-mlflow.json"))
    to_json_tensorflow_serving(x_test, os.path.join(args.output_dir,f"{args.base_name}-tf_serving.json"))
