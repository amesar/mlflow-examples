package org.andre.mlflow.examples.wine.sparkml

import java.io.{File,PrintWriter}
import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.{SparkSession,DataFrame}
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.mlflow.tracking.{MlflowClient,MlflowClientVersion}
import org.mlflow.api.proto.Service.RunStatus
import org.andre.mlflow.util.MLflowUtils
import org.andre.mleap.util.SparkBundleUtils
import org.andre.mlflow.examples.wine.WineUtils

/**
 * MLflow DecisionTreeRegressor with wine quality data.
 */
object TrainWine {
  val spark = SparkSession.builder.appName("DecisionTreeRegressionExample").getOrCreate()
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
    println(s"  maxBins: ${opts.maxBins}")
    println(s"  runOrigin: ${opts.runOrigin}")
    println(s"  skipMLeap: ${opts.skipMLeap}")

    // MLflow - create or get existing experiment
    val client = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)

    val experimentId = MLflowUtils.getOrCreateExperimentId(client, opts.experimentName)
    println("Experiment ID: "+experimentId)

    // Train model
    train(client, experimentId, opts.modelPath, opts.maxDepth, opts.maxBins, opts.runOrigin, opts.dataPath, opts.skipMLeap)
  }

  def train(client: MlflowClient, experimentId: String, modelDir: String, maxDepth: Int, maxBins: Int, runOrigin: String, dataPath: String, skipMLeap: Boolean) {

    // Read data
    val data = WineUtils.readData(spark, dataPath)

    // Process data
    println("Input data count: "+data.count())
    val columns = data.columns.toList.filter(_ != WineUtils.colLabel)
    val assembler = new VectorAssembler()
      .setInputCols(columns.toArray)
      .setOutputCol("features")
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 2019)

    // MLflow - create run
    val runInfo = client.createRun(experimentId)
    val runId = runInfo.getRunId()
    println(s"Run ID: $runId")
    println(s"runOrigin: $runOrigin")
    println(s"SparkVersion: ${spark.version}")
    println(s"ScalaVersion: ${util.Properties.versionString}")
    println(s"MLeapVersion: ${SparkBundleUtils.getMLeapBundleVersion}")

    // MLflow - set tags
    client.setTag(runId, "dataPath",dataPath)
    client.setTag(runId, "mlflow.source.name",MLflowUtils.getSourceName(getClass()))
    client.setTag(runId, "mlflowVersion",MlflowClientVersion.getClientVersion())
    client.setTag(runId, "sparkVersion",spark.version)
    client.setTag(runId, "scalaVersion",util.Properties.versionString)
    client.setTag(runId, "MLeapVersion",SparkBundleUtils.getMLeapBundleVersion)

    // MLflow - log parameters
    val params = Seq(("maxDepth",maxDepth),("maxBins",maxBins),("runOrigin",runOrigin))
    println(s"Params:")
    for (p <- params) {
      println(s"  ${p._1}: ${p._2}")
      client.logParam(runId, p._1,p._2.toString)
    }

    // Create model
    val dt = new DecisionTreeRegressor()
      .setLabelCol(WineUtils.colLabel)
      .setFeaturesCol(WineUtils.colFeatures)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)

    // Create pipeline
    val pipeline = new Pipeline().setStages(Array(assembler,dt))

    // Fit model
    val model = pipeline.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)
    println("Predictions Schema:")

    // MLflow - log metrics
    val metrics = Seq("rmse","r2", "mae")
    println("Metrics:")
    for (metric <- metrics) {
      val evaluator = new RegressionEvaluator()
        .setLabelCol(WineUtils.colLabel)
        .setPredictionCol(WineUtils.colPrediction)
        .setMetricName(metric)
      val v = evaluator.evaluate(predictions)
      println(s"  $metric: $v - isLargerBetter: ${evaluator.isLargerBetter}")
      client.logMetric(runId, metric, v)
    }

    // MLflow - log tree model artifact
    val treeModel = model.stages.last.asInstanceOf[DecisionTreeRegressionModel]
    val path = "treeModel.txt"
    new PrintWriter(path) { write(treeModel.toDebugString) ; close }
    client.logArtifact(runId, new File(path),"details")

    // MLflow - Save model in Spark ML and MLeap formats
    logModelAsSparkML(client, runId, modelDir, model)
    if (!skipMLeap) {
        logModelAsMLeap(client, runId, modelDir, model, data, predictions)
    }

    // MLflow - close run
    client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }

  def logModelAsSparkML(client: MlflowClient, runId: String, modelDir: String, model: PipelineModel) = {
    val modelPath = s"$modelDir/spark-model"
    model.write.overwrite().save(modelPath)
    val artifactPath = "spark-model/sparkml" // NOTE: compatible with Python SparkML path convention
    client.logArtifacts(runId, new File(modelPath), artifactPath)
  }
  
  def logModelAsMLeap(client: MlflowClient, runId: String, modelDir: String, model: PipelineModel, data: DataFrame, predictions: DataFrame) {
    // Print schemas
    println("Data Schema: ")
    data.printSchema
    println("Predictions Schema: ")
    predictions.printSchema

    // Log model as MLeap artifact
    val modelPath = new File(s"$modelDir/mleap-model")
    modelPath.mkdir
    SparkBundleUtils.saveModel(s"file:${modelPath.getAbsolutePath}", model, predictions)
    client.logArtifacts(runId, modelPath, "mleap-model/mleap/model") // Make compatible with MLflow Python mlflow.mleap.log_model path

    // Log mleap schema file for MLeap runtime deserialization
    val schemaPath = new File("schema.json")
    new java.io.PrintWriter(schemaPath) { write(data.schema.json) ; close }
    client.logArtifact(runId, schemaPath, "mleap-model")
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
    var maxDepth: Int = 5 // per doc

    @Parameter(names = Array("--maxBins" ), description = "maxBins param", required=false)
    var maxBins: Int = 32 // per doc

    @Parameter(names = Array("--runOrigin" ), description = "runOrigin tag", required=false)
    var runOrigin = "None"

    @Parameter(names = Array("--experimentName" ), description = "Experiment name", required=false)
    var experimentName = "scala_classic"

    @Parameter(names = Array("--skipMLeap" ), description = "Score with MLeap also", required=false)
    var skipMLeap: Boolean = false
  }
}
