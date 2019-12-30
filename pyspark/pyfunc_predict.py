# Serve predictions with mlflow.pyfunc.load_pyfunc()

import sys
import pandas as pd
import mlflow
import mlflow.pyfunc
print("MLflow Version:", mlflow.version.VERSION)

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI PREDICTION_FILE")
        sys.exit(1)
    model_uri = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../data/wine-quality-white.csv"
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    model = mlflow.pyfunc.load_pyfunc(model_uri)
    print("model:", model)

    data = pd.read_csv(data_path)
    print("data.shape:",data.shape)

    predictions = model.predict(data)
    print("predictions:", predictions[:5])
    print("predictions.len:", len(predictions))

