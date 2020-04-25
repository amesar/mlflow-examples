package org.andre.mleap.util

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._

object SparkBundleUtils {

  def getMLeapBundleVersion = ml.combust.bundle.dsl.Bundle.version

  def saveModel(bundlePath: String, model: PipelineModel, data: DataFrame) {
    val bundle = BundleFile(bundlePath)
    try {
      val context = SparkBundleContext().withDataset(data)
      model.writeBundle.save(bundle)(context)
    } finally {
      bundle.close()
    }
  }

  def readModel(bundlePath: String) = {
    val bundle = BundleFile(bundlePath)
    try {
      bundle.loadSparkBundle().get.root
    } finally {
      bundle.close()
    }
  } 
}
