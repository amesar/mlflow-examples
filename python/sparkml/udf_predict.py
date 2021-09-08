import click
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from common import *

spark = SparkSession.builder.appName("Predict").getOrCreate()
show_versions(spark)

@click.command()
@click.option("--model-uri", help="Model URI", required=True, type=str)
@click.option("--data-path", help="Data path", default=default_data_path, type=str)

def main(model_uri, data_path):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    data = read_data(spark, data_path)

    print("model_uri:",model_uri)
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    print("==== UDF - workaround")
    predictions = data.withColumn("prediction", udf(*data.columns))
    print("predictions.type:", type(predictions))
    df = predictions.select(colLabel, colPrediction)
    df.show(5,False)

if __name__ == "__main__":
    main()
