import sys
import pandas as pd
import mlflow
import mlflow.pyfunc
import utils

print("MLflow Version:", mlflow.version.VERSION)

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("ERROR: Expecting MODEL_URI")
        sys.exit(1)
    model_uri = sys.argv[1]
    print("model_uri:", model_uri)
    model = mlflow.pyfunc.load_pyfunc(model_uri)
    print("model:", model)

    _,_,ndarray,_  = utils.build_data()
    data = pd.DataFrame(ndarray)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)
