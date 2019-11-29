package org.andre.mlflow.util

import ml.combust.mleap.runtime.frame.{DefaultLeapFrame,Transformer}

object PredictUtils {
  def predict(model: Transformer, data: DefaultLeapFrame) {
    println(s"Model class: ${model.getClass.getName}")

    val transformed = model.transform(data).get
    val predictions = transformed.select("prediction").get.dataset.map(p => p.getDouble(0))

    val sum = predictions.sum
    println(f"Prediction sum: ${sum}%.3f")

    val groups = predictions.groupBy(x => x).mapValues(_.size).toSeq.sortBy(_._2).reverse
    println(s"Prediction Counts:")
    println(s"  prediction    count")
    for (g <- groups) {
      println(f"     ${g._1}%7.3f ${g._2}%8d")
    }
    println(s"${predictions.size} Predictions:")
    for (p <- predictions.take(10)) {
      println(f"  $p%7.3f")
    }
  }
}
