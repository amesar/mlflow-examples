# Serve predictions with ONNX flavor

import sys
import numpy
import mlflow
import mlflow.onnx
import utils
import mlflow
import onnxruntime as rt

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../data/wine-quality-white.csv"
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    data = utils.read_prediction_data(data_path)
    data = data.to_numpy()

    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))

    # Score with ONNX runtime
    sess = rt.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    predictions = sess.run(None, {input_name: data.astype(numpy.float32)})[0]

    utils.display_predictions(predictions)
