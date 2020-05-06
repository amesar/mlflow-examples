# Serve predictions with mlflow.pyfunc.load_model()

import sys
import mlflow
import mlflow.pyfunc
import utils

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.__version__)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../../data/train/wine-quality-white.csv"
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    client = mlflow.tracking.MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri)
    print("model:", model)

    data = utils.read_prediction_data(data_path)
    predictions = model.predict(data)
    print("predictions:\n", predictions)
