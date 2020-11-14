# Serve predictions with mlflow.pyfunc.load_model()

import sys
import mlflow
import mlflow.pyfunc
import predict_utils

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../../data/train/wine-quality-white.csv"
    print("Arguments:")
    print("  data_path:", data_path)
    print("  model_uri:", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    print("model:", model)
    print("model.metadata:", model.metadata)
    print("model.metadata.type:", type(model.metadata))

    data = predict_utils.read_prediction_data(data_path)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)
