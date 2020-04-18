package org.andre.mlflow.examples.wine

import org.apache.spark.sql.SparkSession

object WineUtils {
  val colLabel = "quality"
  val colPrediction = "prediction"
  val colFeatures = "features"

  def readData(spark: SparkSession, dataPath: String) = {
    if (dataPath.endsWith(".csv")) {
      spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(dataPath)
    } else if (dataPath.endsWith(".parquet")) {
      spark.read.format("parquet").load(dataPath)
    } else {
      throw new Exception("Input data format not supported: $dataPath")
    }
  }
}
