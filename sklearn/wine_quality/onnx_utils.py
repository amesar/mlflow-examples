import numpy as np
import mlflow
import mlflow.onnx
import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def log_model(model, artifact_path, data):
    initial_type = [('float_input', FloatTensorType([None, data.shape[0]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    print("onnx_model.type:",type(onnx_model))
    mlflow.set_tag("onnx_version",onnx.__version__)
    mlflow.onnx.log_model(onnx_model, artifact_path)
    return onnx_model

def score(model, data_ndarray):
    sess = rt.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: data_ndarray.astype(np.float32)})[0]
