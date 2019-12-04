from argparse import ArgumentParser
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
#from pyspark.ml import PipelineModel
#from mleap.pyspark.spark_support import SimpleSparkSerializer
import mleap_utils
from common import *

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="run_id", required=True)
    parser.add_argument("--data_path", dest="data_path", help="data_path", required=False)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))

    spark = SparkSession.builder.appName("Predict").getOrCreate()
    data_path = args.data_path or default_data_path
    data = read_data(spark, data_path)
    print("Data Schema:")
    data.printSchema()

    # Predict with Spark ML
    print("Spark ML predictions")
    model_uri = f"runs:/{args.run_id}/spark-model"
    print("model_uri:",model_uri)
    model = mlflow.spark.load_model(model_uri)
    predictions = model.transform(data)
    df = predictions.select(colPrediction, colLabel, colFeatures)
    df.show(5,False)

    # Predict with MLeap as SparkBundle
    print("MLeap ML predictions")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(args.run_id)
    model = mleap_utils.read_model(run, "mleap-model/mleap/model")
    predictions = model.transform(data)
    df = predictions.select(colPrediction, colLabel, colFeatures)
    df.show(5,False)
