import numpy as np
import onnx
import onnxruntime 
import mlflow

def log_model(model, artifact_path):
    onnx_model = onnxmltools.convert_keras(model, artifact_path)
    print("onnx_model.type:",type(onnx_model))
    mlflow.onnx.log_model(onnx_model, artifact_path)
    mlflow.set_tag("onnx_version",onnx.__version__)

def score(model, data):
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    return sess.run(None, {input_name: data.astype(np.float32)})[0]
