package org.andre.mlflow.util

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._

object MLeapUtils {

  def saveModelAsSparkBundle(bundlePath: String, model: PipelineModel, data: DataFrame) {
    val bundle = BundleFile(bundlePath)
    try {
      val context = SparkBundleContext().withDataset(data)
      model.writeBundle.save(bundle)(context)
    } finally {
      bundle.close()
    }
  }

  def readModelAsSparkBundle(bundlePath: String) = {
    val bundle = BundleFile(bundlePath)
    try {
      bundle.loadSparkBundle().get.root
    } finally {
      bundle.close()
    }
  } 
}
