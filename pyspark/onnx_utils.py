import numpy as np
import mlflow.onnx
import onnx
import onnxmltools
import onnxruntime
from onnxmltools.convert.common.data_types import FloatTensorType

def log_model(spark, model, name, data):
    initial_types = [ (col, FloatTensorType([1, 1])) for col in data.columns ]
    onnx_model = onnxmltools.convert_sparkml(model, name, initial_types, spark_session=spark)
    mlflow.onnx.log_model(onnx_model, "onnx-model")
    mlflow.set_tag("onnx_version", onnx.__version__)

def score_model(model, data):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    input_feed = { col.name: data.astype(np.float32) for col in sess.get_inputs() }
    return sess.run(None, input_feed)[0]
