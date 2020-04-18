package org.andre.mlflow.examples.wine.sparkml

import com.beust.jcommander.{JCommander, Parameter}
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils
import org.andre.mleap.util.MLeapBundleUtils

/**
 * Loads model as MLeapBundle with no Spark dependencies.
 */
object PredictAsMLeapBundle {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  runId: ${opts.runId}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val schemaPath = client.downloadArtifacts(opts.runId,"mleap-model/schema.json").getAbsolutePath
    println("schemaPath: "+schemaPath)
    val schema = MLeapBundleUtils.readSchema(schemaPath)

    val records = readData(opts.dataPath)
    val data = DefaultLeapFrame(schema, records)

    val modelPath = client.downloadArtifacts(opts.runId,"mleap-model/mleap/model").getAbsolutePath
    val bundleUri = s"file:${modelPath}"

    val model = MLeapBundleUtils.readModel(bundleUri)
    val transformed = model.transform(data).get
    val predictions = transformed.select("prediction").get.dataset.map(p => p.getDouble(0))

    showSummary(predictions)
    showGroupByCounts(predictions)
  }

  def readData(dataPath: String) = {
    import scala.io.Source
    val lines = Source.fromFile(dataPath).getLines.toSeq.drop(1) // skip header
    val data =  lines.map(x => x.split(",").toSeq ).toSeq
    data.map(x => Row(x(0).toDouble,x(1).toDouble,x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toDouble, x(7).toDouble, x(8).toDouble, x(9).toDouble, x(10).toDouble,x(11).toInt))
  }

  def showSummary(predictions: Seq[Double]) {
    val sum = predictions.sum
    println(f"Prediction count: ${predictions.size}")
    println(f"Prediction sum:   ${sum}%.3f")
  }

  def showGroupByCounts(predictions: Seq[Double]) {
    val groups = predictions.groupBy(x => x).mapValues(_.size).toSeq.sortBy(_._2).reverse
    println(s"  prediction    count")
    for (g <- groups.take(10)) { 
      println(f"     ${g._1}%7.3f ${g._2}%8d")
    } 
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
