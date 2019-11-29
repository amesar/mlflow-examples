package org.andre.mlflow.util

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.core.types.StructType
import ml.combust.mleap.json.JsonSupport._
import ml.combust.bundle.dsl.Bundle
import resource.managed
import spray.json._
import scala.io.Source

object MLeapUtils {
  println("Mleap Bundle version: "+Bundle.version)

  def readModel(bundlePath: String) = {
    val bundle = BundleFile(bundlePath)
    try {
      bundle.loadMleapBundle.get.root
    } finally {
      bundle.close()
    }
  }

  /** Read JSON Spark schema and create MLeap schema. */
  def readSchema(path: String) : StructType = {
    val json = Source.fromFile(path).mkString
    json.parseJson.convertTo[StructType]
  }
}
