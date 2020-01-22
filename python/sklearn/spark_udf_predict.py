"""
Serve predictions with Spark UDF.
"""

import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.pyfunc

if __name__ == "__main__":
    model_uri = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "../../data/wine-quality-white.csv"
    print("path:", path)
    print("model_uri:", model_uri)
    print("MLflow Version:", mlflow.version.VERSION)

    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()

    data = spark.read.option("inferSchema",True).option("header",True).csv(path) if path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(path)

    if "quality" in data.columns:
        data = data.drop("quality")
    data.show(10)

    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = data.withColumn("prediction", udf(*data.columns))
    predictions.show(10)
    predictions.select("prediction").show(10)
    pred = predictions.select("prediction").first()[0]
    print("predictions: {:,.7f}".format(pred))
