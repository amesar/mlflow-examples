# Serve predictions with ONNX flavor

import sys
import mlflow
import mlflow.onnx
import utils
import onnx_utils

print("MLflow Version:", mlflow.__version__)

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../../data/train/wine-quality-white.csv"
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    data = utils.read_prediction_data(data_path)
    data = data.to_numpy()

    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    predictions = onnx_utils.score(model, data)
    utils.display_predictions(predictions)
