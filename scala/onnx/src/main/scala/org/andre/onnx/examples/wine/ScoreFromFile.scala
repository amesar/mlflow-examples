package org.andre.onnx.examples.wine

import com.beust.jcommander.{JCommander, Parameter}

object ScoreFromFile {
  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  modelPath: ${opts.modelPath}")
    val predictions = OnnxScorer.predict(opts.modelPath, opts.dataPath)
    println("Predictions:")
    for (p <- predictions) {
      println("  "+p)
    }
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path")
    var dataPath = "../../data/wine-quality-white.csv"

    @Parameter(names = Array("--modelPath" ), description = "Data path", required=true)
    var modelPath: String = null
  }
}
