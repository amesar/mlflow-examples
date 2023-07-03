// Databricks notebook source
// MAGIC %md ## MLflow Scala train and Scala score
// MAGIC * Train and log model in Scala and then score it in Scala.
// MAGIC * Uses the classic MLflow Java API.
// MAGIC * TODO: Score in Python.
// MAGIC
// MAGIC Last updated: 2023-07-03

// COMMAND ----------

// MAGIC %md ### Setup

// COMMAND ----------

// MAGIC %run ./Common_Scala

// COMMAND ----------

import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.mlflow.api.proto.Service.RunStatus

// COMMAND ----------

//dbutils.widgets.removeAll()

// COMMAND ----------

val defaultExperiment = s"$workspaceHome/SparkML_Scala"

dbutils.widgets.text("Experiment", defaultExperiment)
dbutils.widgets.text("maxDepth", "2")
dbutils.widgets.text("maxBins", "32")
val experimentName = dbutils.widgets.get("Experiment")
val maxDepth = dbutils.widgets.get("maxDepth").toInt
val maxBins = dbutils.widgets.get("maxBins").toInt

// COMMAND ----------

// MAGIC %md ### Get Data

// COMMAND ----------

val data = getData()
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), 42)

// COMMAND ----------

// MAGIC %md ### MLflow Setup

// COMMAND ----------

import org.mlflow.tracking.MlflowClient
val client = new MlflowClient()
val experimentId = getOrCreateExperimentId(client, experimentName)
experimentName

// COMMAND ----------

// MAGIC %md ### Train

// COMMAND ----------

val colLabel = "quality"
val colPrediction = "prediction"
val colFeatures = "features"
val metrics = Seq("rmse","r2", "mae")

// COMMAND ----------

// Create MLflow run
val runInfo = client.createRun(experimentId)
val runId = runInfo.getRunId()
println(s"Run ID: $runId")

// Need this temp dir to log artifacts
val baseScratchDir = s"dbfs:/tmp/mlflow_scratch/run_$runId"
dbutils.fs.rm(baseScratchDir, true)

// Log MLflow tags
client.setTag(runId, "dataPath", dataPath)
client.setTag(runId, "mlflowVersion", MlflowClientVersion.getClientVersion())

// Log MLflow parameters
client.logParam(runId, "maxDepth", maxDepth.toString)
client.logParam(runId, "maxBins", maxBins.toString)
println("Parameters:")
println(s"  maxDepth: $maxDepth")
println(s"  maxBins: $maxBins")

// Create model
val dt = new DecisionTreeRegressor()
  .setLabelCol(colLabel)
  .setFeaturesCol(colFeatures)
  .setMaxBins(maxBins)
  .setMaxDepth(maxDepth)

// Create pipeline
val columns = data.columns.toList.filter(_ != colLabel)
val assembler = new VectorAssembler()
  .setInputCols(columns.toArray)
  .setOutputCol(colFeatures)
val pipeline = new Pipeline().setStages(Array(assembler,dt))

// Fit model
val model = pipeline.fit(trainingData)

// Log MLflow training metrics
val predictions = model.transform(testData)
println("Metrics:")
for (metric <- metrics) { 
  val evaluator = new RegressionEvaluator()
    .setLabelCol(colLabel)
    .setPredictionCol(colPrediction)
    .setMetricName(metric)
  val v = evaluator.evaluate(predictions)
  println(s"  $metric: $v - isLargerBetter: ${evaluator.isLargerBetter}")
  client.logMetric(runId, metric, v)
} 

// Log MLflow model
val modelScratchDir1 = baseScratchDir + "_model_01"
model.write.overwrite().save(modelScratchDir1)
client.logArtifacts(runId, new java.io.File(mkLocalPath(modelScratchDir1)), "spark-model")

// End MLflow run
client.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())

// COMMAND ----------

displayRunUri(experimentId,runId)

// COMMAND ----------

// MAGIC %md ### Predict

// COMMAND ----------

val localModelPath = client.downloadArtifacts(runId,"spark-model")

// COMMAND ----------

val modelScratchDir2 = baseScratchDir + "_model_02"
copyDirectory(localModelPath, new File(mkLocalPath(modelScratchDir2)))

// COMMAND ----------

val model = PipelineModel.load(modelScratchDir2)
val predictions = model.transform(data)
val df = predictions.select(colPrediction, colLabel, colFeatures)
display(df)
