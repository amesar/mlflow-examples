// Databricks notebook source
import org.mlflow.tracking.{MlflowClient,MlflowClientVersion}
println("MLflow Java version: "+MlflowClientVersion.getClientVersion())

// COMMAND ----------

// Workspace paths

val user = dbutils.notebook.getContext().tags("user")
val notebookPath = dbutils.notebook.getContext.notebookPath.get
val _path = new java.io.File(notebookPath)

//val notebookName = notebookPa.split("/").last
val workspaceHome = s"/Users/${user}"

//val experimentName = s"${workspaceHome}/gls_scala_${notebookName}"

// COMMAND ----------

val dataPath = "dbfs:/databricks-datasets/wine-quality/winequality-white.csv"

def getData() = {
  val _data = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .option("delimiter", ";")
    .load(dataPath)

  val columns = _data.columns.map(x => x.replace(" ","_")) // Replace spaces in column names with underscores
  _data.toDF(columns: _*)
}

// COMMAND ----------

// Return the ID of an experiment - create it if it doesn't exist

def getOrCreateExperimentId(client: MlflowClient, experimentName: String) = {
  try { 
    client.createExperiment(experimentName)
  } catch { 
    case e: org.mlflow.tracking.MlflowHttpException => { // statusCode 400
      client.getExperimentByName(experimentName).get.getExperimentId
    } 
  } 
} 

// COMMAND ----------

//  Display the URI of the run in the MLflow UI

def displayRunUri(experimentId: String, runId: String) = {
  val hostName = dbutils.notebook.getContext().tags.get("browserHostName").get
  val uri = s"https://$hostName/#mlflow/experiments/$experimentId/runs/$runId"
  displayHTML(s"""<b>Run URI:</b> <a href="$uri">$uri</a>""")
}

// COMMAND ----------

def mkLocalPath(path: String) = path.replace("dbfs:","/dbfs")
def mkDbfsPath(path: String) = path.replace("/dbfs","dbfs:")

// COMMAND ----------

import java.io.File
import org.apache.commons.io.FileUtils

def _copyDirectory(srcDir: String, dstDir: String) {
  FileUtils.copyDirectory(new File(srcDir), new File(dstDir))
}

def copyDirectory(srcDir: File, dstDir: File) {
  FileUtils.copyDirectory(srcDir, dstDir)
}

// COMMAND ----------


