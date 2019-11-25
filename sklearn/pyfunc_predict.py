# Serve predictions with mlflow.pyfunc.load_pyfunc()

import sys
import mlflow
import mlflow.pyfunc
import util

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../data/wine-quality-white.csv"
    print("data_path:",data_path)
    print("model_uri:",model_uri)

    client = mlflow.tracking.MlflowClient()
    model = mlflow.pyfunc.load_pyfunc(model_uri)
    print("model:",model)

    data = util.read_prediction_data(data_path)
    predictions = model.predict(data)
    print("predictions:",predictions)
