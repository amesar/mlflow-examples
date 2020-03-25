import mlflow
import mlflow.sklearn
import common

print(f"MLflow Version: {mlflow.__version__}")

def score(model_uri, data_path):
    X_train, X_test, y_train, y_test = common.build_data(data_path)
    data = X_test

    print("==== sklearn score")
    model = mlflow.sklearn.load_model(model_uri)
    print("model:", model)
    print("model.type:", type(model))
    predictions = model.predict(data)
    print("predictions:", predictions)

    print("==== pyfunc score")
    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)
    print("model.type:", type(model))
    predictions = model.predict(data)
    print("predictions:", predictions)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_uri", dest="model_uri", help="Model URI", default=common.model_uri)
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=common.data_path)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    score(args.model_uri, args.data_path)
