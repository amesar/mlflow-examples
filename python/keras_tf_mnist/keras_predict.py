import click
import mlflow
import mlflow.keras
import utils

utils.display_versions()

@click.command()
@click.option("--model_uri", help="Model URI", required=True, type=str)
def main(model_uri):
    print("model_uri:", model_uri)
    data = utils.get_prediction_data()
    print("data.type:", type(data))
    print("data.shape:", data.shape)

    print("\n**** mlflow.keras.load_model\n")
    model = mlflow.keras.load_model(model_uri)
    print("model:", type(model))

    print("== model.predict")
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

    print("== model.predict_classes")
    predictions = model.predict_classes(data)
    print("predictions:", predictions)

    utils.predict_pyfunc(model_uri, data)

if __name__ == "__main__":
    main()
