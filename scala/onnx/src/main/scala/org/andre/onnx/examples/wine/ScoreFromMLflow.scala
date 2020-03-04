package org.andre.onnx.examples.wine

import com.beust.jcommander.{JCommander, Parameter}
import org.mlflow.tracking.MlflowClient
import org.andre.mlflow.util.MLflowUtils

object ScoreFromMLflow {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  runId: ${opts.runId}")
    println(s"  artifactPath: ${opts.artifactPath}")
    println(s"  dataPath: ${opts.dataPath}")
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val modelPath = client.downloadArtifacts(opts.runId, opts.artifactPath).getAbsolutePath
    println("modelPath:"+modelPath)
    val predictions = OnnxScorer.predict(modelPath, opts.dataPath)
    println("Predictions:")
    for (p <- predictions) {
      println("  "+p)
    }
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path")
    var dataPath = "../../data/wine-quality-white.csv"

    @Parameter(names = Array("--artifactPath" ), description = "Artifact path", required=true)
    var artifactPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "Databricks REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "Run ID", required=true)
    var runId: String = null
  }
}
