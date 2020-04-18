package org.andre.mlflow.examples.wine.xgboost

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils
import org.andre.mlflow.examples.wine.WineUtils

object Predict {
  val colLabel2 = "classIndex"

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  runId: ${opts.runId}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = WineUtils.readData(spark, opts.dataPath)
    println("data.schema:") ; data.printSchema

    val modelPath = client.downloadArtifacts(opts.runId, "xgboost-model").getAbsolutePath
    val model = XGBoostRegressionModel.load(modelPath)
    println("model.class: "+model.getClass)

    val dataTransformed = Utils.transformData(data)
    println("dataTransformed.schema:") ; dataTransformed.printSchema

    val predictions = model.transform(dataTransformed)
    println("predictions.schema:") ; predictions.printSchema
    predictions.show(10)
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "Databricks REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "Run ID", required=true)
    var runId: String = null
  }
}
