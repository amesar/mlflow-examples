package org.andre.mlflow.examples.wine

import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame,Row}
import org.andre.mlflow.util.{MLflowUtils,MLeapUtils,PredictUtils}

object PredictWine {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    MLflowUtils.showVersion()
    println("Options:")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  runId: ${opts.runId}")
    println(s"  autoSchema: ${opts.autoSchema}")

    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val modelPath = "file:" + client.downloadArtifacts(opts.runId, "mleap-model/mleap/model").getAbsolutePath
    val model = MLeapUtils.readModel(modelPath)

    val schema = if (opts.autoSchema) {
      model.inputSchema // NOTE: this croaks
    } else {
      val schemaPath = client.downloadArtifacts(opts.runId, "mleap-model/schema.json").getAbsolutePath
      println("schemaPath: "+schemaPath)
      MLeapUtils.readSchema(schemaPath)
    }
    println("Schema: "+schema)
    val records = readData(opts.dataPath)
    val data = DefaultLeapFrame(schema, records)

    PredictUtils.predict(model, data)
  }

  def readData(dataPath: String) = {
    import scala.io.Source
    val lines = Source.fromFile(dataPath).getLines.toSeq.drop(1)
    val lst =  lines.map(x => x.split(",").toSeq ).toSeq
    lst.map(x => Row(x(0).toDouble,x(1).toDouble,x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble, x(6).toDouble, x(7).toDouble, x(8).toDouble, x(9).toDouble, x(10).toDouble,x(11).toInt))
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

    @Parameter(names = Array("--autoSchema" ), description = "autoSchema", required=false)
    var autoSchema = false
  }
}
