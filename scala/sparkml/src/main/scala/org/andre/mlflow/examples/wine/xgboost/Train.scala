package org.andre.mlflow.examples.wine.xgboost

import java.io.File
import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.tracking.{MlflowClient,MlflowClientVersion}
import org.mlflow.api.proto.Service.RunStatus
import org.andre.mlflow.util.MLflowUtils
import org.andre.mlflow.examples.wine.WineUtils

object Train {
  val spark = SparkSession.builder.appName("Train").getOrCreate()
  MLflowUtils.showVersions(spark)

  def main(args: Array[String]) {
    new JCommander(opts, args: _*)
    println("Options:")
    println(s"  trackingUri: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  experimentName: ${opts.experimentName}")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  modelPath: ${opts.modelPath}")
    println(s"  maxDepth: ${opts.maxDepth}")

    // MLflow - create or get existing experiment
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val experimentId = MLflowUtils.getOrCreateExperimentId(client, opts.experimentName)
    println("Experiment ID: "+experimentId)

    // Train model
    train(client, experimentId, opts.modelPath, opts.maxDepth, opts.objective, opts.dataPath)
  }

  def train(client: MlflowClient, experimentId: String, modelPath: String, maxDepth: Int, objective: String, dataPath: String) {
    // Read data
    val data = WineUtils.readData(spark, dataPath)

    // MLflow - create run
    val runInfo = client.createRun(experimentId)
    val runId = runInfo.getRunId()
    println(s"Run ID: $runId")
    println(s"sparkVersion: ${spark.version}")
    println(s"scalaVersion: ${util.Properties.versionString}")

    // MLflow - set tags
    client.setTag(runId, "dataPath",dataPath)
    client.setTag(runId, "mlflow.source.name",MLflowUtils.getSourceName(getClass()))
    client.setTag(runId, "mlflowVersion",MlflowClientVersion.getClientVersion())
    client.setTag(runId, "sparkVersion",spark.version)
    client.setTag(runId, "scalaVersion",util.Properties.versionString)

    // Process data
    val dataTransformed = Utils.transformData(data)
    println("dataTransformed.schema: "+ dataTransformed.schema.treeString)
    val Array(train, eval1, eval2, test) = dataTransformed.randomSplit(Array(0.6, 0.2, 0.1, 0.1))

    // Parameters
    val params = Map("eta" -> "1", 
      "max_depth" -> maxDepth, 
      "objective" -> "rank:pairwise",
      "num_round" -> 5, 
      "num_workers" -> 2, 
      "objective" -> objective,
      "eval_sets" -> Map("eval1" -> eval1, "eval2" -> eval2))
    println("Parameters:")
    for ((k,v) <- params) {
      println(s"  $k: $v")
      client.logParam(runId, k, v.toString)
    }

    val xgb = new XGBoostRegressor(params)
      .setFeaturesCol(WineUtils.colFeatures)
      .setLabelCol(Utils.colOutput)
    val model = xgb.fit(train)
    val predictions = model.transform(test)
    println("Predictions:")
    predictions.show(5)

    // MLflow - log metrics
    val metrics = Seq("rmse","r2", "mae")
    println("Metrics:")
    for (metric <- metrics) {
      val evaluator = new RegressionEvaluator()
        .setLabelCol(Utils.colOutput)
        .setPredictionCol(WineUtils.colPrediction)
        .setMetricName(metric)
      val v = evaluator.evaluate(predictions)
      println(s"  $metric: $v - isLargerBetter: ${evaluator.isLargerBetter}")
      client.logMetric(runId, metric, v)
    }

    // MLflow - log model
    val path = s"$modelPath/spark-model"
    model.write.overwrite().save(path)
    client.logArtifacts(runId, new File(path), "xgboost-model")

    // MLflow - close run
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }

  object opts {
    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--modelPath" ), description = "Model path", required=true)
    var modelPath: String = null

    @Parameter(names = Array("--maxDepth" ), description = "maxDepth param", required=false)
    var maxDepth: Int = 6 // per doc

    @Parameter(names = Array("--objective" ), description = "objective param", required=false)
    var objective = "reg:squarederror" // per doc

    @Parameter(names = Array("--experimentName" ), description = "Experiment name", required=false)
    var experimentName = "scala_classic"
  }
}
