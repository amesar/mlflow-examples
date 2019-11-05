"""
Serve predictions with Spark UDF.
"""

import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 else "../data/wine-quality-white.csv"
    run_id = sys.argv[1]
    print("path:",path)
    print("run_id=",run_id)
    print("MLflow Version:", mlflow.version.VERSION)

    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()

    df = spark.read.option("inferSchema",True).option("header", True).csv(path) if path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(path)

    if "quality" in df.columns:
        df = df.drop("quality")
    df.show(10)

    model_uri = f"runs:/{run_id}/sklearn-model"
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    df2 = df.withColumn("prediction", udf(*df.columns))
    df2.show(10)
    df2.select("prediction").show(10)
    pred = df2.select("prediction").first()[0]
    print("predictions: {:,.7f}".format(pred))

