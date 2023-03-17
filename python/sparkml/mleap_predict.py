import click
import mlflow
##import mlflow.spark
from pyspark.sql import SparkSession
##from common import *
from common import show_versions, read_data, default_data_path
from common import colLabel, colPrediction, colFeatures
import mleap_utils

spark = SparkSession.builder.appName("Predict").getOrCreate()
show_versions(spark)

@click.command()
@click.option("--run-id", help="Run ID", required=True, type=str)
@click.option("--data-path", help="Data path", default=default_data_path, type=str)

def main(run_id, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    #model_uri = f"runs:/{run_id}/mleap-model"

    data_path = data_path or default_data_path
    data = read_data(spark, data_path)
    print("Data Schema:")
    data.printSchema()

    print("MLeap predictions")
    client = mlflow.client.MlflowClient()
    run = client.get_run(run_id)
    model = mleap_utils.load_model(run, "mleap-model/mleap/model")
    print("model.type:", type(model))
    predictions = model.transform(data)
    print("predictions.type:", type(predictions))
    df = predictions.select(colPrediction, colLabel, colFeatures)
    df.show(5, False)

if __name__ == "__main__":
    main()
