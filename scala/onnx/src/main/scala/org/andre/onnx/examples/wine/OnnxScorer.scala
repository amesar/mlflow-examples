package org.andre.onnx.examples.wine

import java.nio.file.{Files, Paths}
import scala.io.Source
import ai.onnxruntime.{OrtEnvironment, OrtSession, OnnxTensor}
import scala.collection.JavaConversions._

object OnnxScorer {
  def predict(modelPath: String, dataPath: String) : Array[Float] = {
    val src = Source.fromFile(dataPath).getLines
    val header = src.take(1).next
    val data = src.map(line => line.split(",").toArray.map(x => x.toFloat).dropRight(1) )

    val env = OrtEnvironment.getEnvironment("WineQuality")
    val bytes = Files.readAllBytes(Paths.get(modelPath))
    val session = env.createSession(bytes, new OrtSession.SessionOptions())
    val inputMap = Map("float_input" -> OnnxTensor.createTensor(env, data.toArray))

    val res = session.run(mapAsJavaMap(inputMap), session.getOutputNames())
    val tensor = res.get(0)
    val predictions = tensor.getValue.asInstanceOf[Array[Array[Float]]]
    predictions.map(x => x(0))
  }
}
