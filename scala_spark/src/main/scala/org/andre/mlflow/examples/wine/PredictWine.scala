package org.andre.mlflow.examples.wine

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{PipelineModel,Transformer}
import org.apache.spark.sql.functions.desc
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.{MLflowUtils,MLeapUtils}

object PredictWine {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  runId: ${opts.runId}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = Utils.readData(spark, opts.dataPath)

    println("==== Spark ML")
    val modelPath0 = client.downloadArtifacts(opts.runId, "spark-model").getAbsolutePath
    val model0 = PipelineModel.load(modelPath0)
    showPredictions(model0, data)

    println("==== MLeap")
    val modelPath1 = "file:" + client.downloadArtifacts(opts.runId, "mleap-model/mleap/model").getAbsolutePath
    val model1 = MLeapUtils.readModelAsSparkBundle(modelPath1)
    showPredictions(model1, data)
  }

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select(Utils.colFeatures,Utils.colLabel,Utils.colPrediction).sort(Utils.colFeatures,Utils.colLabel,Utils.colPrediction)
    df.show(10)
    predictions.groupBy("prediction").count().sort(desc("count")).show
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "runId", required=true)
    var runId: String = null
  }
}
