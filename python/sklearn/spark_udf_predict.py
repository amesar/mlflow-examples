"""
Serve predictions with Spark UDF.
"""

import sys
from pyspark.sql import SparkSession
import mlflow
import mlflow.pyfunc

if __name__ == "__main__":
    model_uri = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "../../data/train/wine-quality-white.csv"
    print("path:", path)
    print("model_uri:", model_uri)
    print("MLflow Version:", mlflow.__version__)
    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
    print("Spark Version:", spark.version)

    data = spark.read.option("inferSchema",True).option("header",True).csv(path) if path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(path)
    if "quality" in data.columns:
        data = data.drop("quality")

    print("\nUDF with DataFrame API")
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = data.withColumn("prediction", udf(*data.columns))
    predictions.show(10)

    print("UDF with SQL")
    spark.udf.register("predictUDF", udf)
    data.createOrReplaceGlobalTempView("data")
    predictions = spark.sql("select *, predictUDF(*) as prediction from global_temp.data")
    predictions.show(10)


