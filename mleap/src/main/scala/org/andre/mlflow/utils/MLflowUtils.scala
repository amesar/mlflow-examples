package org.andre.mlflow.util

import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.MlflowHttpException
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.tracking.MlflowClientVersion

object MLflowUtils {

  def showVersion() {
    println("MLflow version: "+MlflowClientVersion.getClientVersion())
  }

  def createMlflowClient(args: Array[String]) = {
    println("args: "+args.toList)
    if (args.length == 0) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      val trackingUri = args(0)
      println(s"Tracking URI: $trackingUri")
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }

  def createMlflowClient(trackingUri: String, token: String = "") = {
    if (trackingUri == null) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      if (token != null) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri, token))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }

}
