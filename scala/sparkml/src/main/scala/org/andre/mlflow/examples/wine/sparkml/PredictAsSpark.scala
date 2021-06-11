package org.andre.mlflow.examples.wine.sparkml

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{PipelineModel,Transformer}
import org.apache.spark.sql.functions.desc
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils
import org.andre.mleap.util.SparkBundleUtils
import org.andre.mlflow.examples.wine.WineUtils

/**
Predicts from Spark ML and MLeap models. Reads from artifact `spark-model` and `mleap-model/mleap/model'.
*/

object PredictAsSpark {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  runId: ${opts.runId}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    MLflowUtils.showVersions(spark)
    val data = WineUtils.readData(spark, opts.dataPath)

    println("==== Spark ML")
    val modelPath = client.downloadArtifacts(opts.runId, "spark-model/sparkml").getAbsolutePath
    val model = PipelineModel.load(modelPath)
    showPredictions(model, data)

    try {
      println("==== MLeap")
      val modelPath = "file:" + client.downloadArtifacts(opts.runId, "mleap-model/mleap/model").getAbsolutePath
      val model = SparkBundleUtils.readModel(modelPath)
      showPredictions(model, data)
    } catch {
      case e: Exception => {
        val emsg = "IOError('No such file or directory"
        if (e.getMessage.contains(emsg)) {
          println("NOTE: MLeap model is not available")
        } else {
          throw e
        }
      }
    }
  }

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select(WineUtils.colFeatures,WineUtils.colLabel,WineUtils.colPrediction).sort(WineUtils.colFeatures,WineUtils.colLabel,WineUtils.colPrediction)
    df.show(10)
    predictions.groupBy("prediction").count().sort(desc("count")).show
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
