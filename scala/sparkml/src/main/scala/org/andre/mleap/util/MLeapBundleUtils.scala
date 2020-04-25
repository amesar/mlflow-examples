package org.andre.mleap.util

import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.core.types.StructType
import ml.combust.mleap.json.JsonSupport._
import ml.combust.bundle.dsl.Bundle
import ml.combust.mleap.runtime.frame.Transformer
import resource.managed
import spray.json._
import scala.io.Source

object MLeapBundleUtils {
  println("Mleap Bundle version: "+Bundle.version)

  def getMLeapBundleVersion = Bundle.version

  def readModel(bundlePath: String) : Transformer = {
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
