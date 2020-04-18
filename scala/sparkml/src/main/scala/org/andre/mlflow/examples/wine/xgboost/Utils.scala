package org.andre.mlflow.examples.wine.xgboost

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer}
import org.andre.mlflow.examples.wine.WineUtils

object Utils {
  val colOutput = "classIndex"

  def transformData(data: DataFrame) : DataFrame = {
    println("Input data count: "+data.count())
    val columns = data.columns.toList.filter(_ != WineUtils.colLabel)
    val assembler = new VectorAssembler()
      .setInputCols(columns.toArray)
      .setOutputCol(WineUtils.colFeatures)
    val indexer = new StringIndexer()
      .setInputCol(WineUtils.colLabel)
      .setOutputCol(colOutput)
      .fit(data)
    val labelTransformed = indexer.transform(data).drop(WineUtils.colLabel)
    assembler.transform(labelTransformed).select(WineUtils.colFeatures, colOutput)
  }
}
