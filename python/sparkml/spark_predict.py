import click
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from common import *

print("MLflow Version:", mlflow.__version__)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

@click.command()
@click.option("--run_id", help="RunID", default=None, type=str)
@click.option("--data_path", help="Data path", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--score_as_mleap", help="Score as MLeap", default=False, type=bool)

def main(run_id, data_path, score_as_mleap):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    spark = SparkSession.builder.appName("Predict").getOrCreate()
    data_path = data_path or default_data_path
    data = read_data(spark, data_path)
    print("Data Schema:")
    data.printSchema()

    # Predict with Spark ML
    print("Spark ML predictions")
    model_uri = f"runs:/{run_id}/spark-model"
    print("model_uri:", model_uri)
    model = mlflow.spark.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.transform(data)
    print("predictions.type:", type(predictions))
    df = predictions.select(colPrediction, colLabel, colFeatures)
    df.show(5, False)

    # Predict with MLeap as SparkBundle
    if score_as_mleap:
        import mleap_utils
        print("MLeap predictions")
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        model = mleap_utils.load_model(run, "mleap-model/mleap/model")
        print("model.type:", type(model))
        predictions = model.transform(data)
        print("predictions.type:", type(predictions))
        df = predictions.select(colPrediction, colLabel, colFeatures)
        df.show(5, False)

if __name__ == "__main__":
    main()
