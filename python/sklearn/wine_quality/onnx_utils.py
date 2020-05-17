import numpy as np
import mlflow
import mlflow.onnx

def log_model(model, artifact_path, model_name, data):
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_types = [('float_input', FloatTensorType([None, data.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types)
    print("onnx_model.type:",type(onnx_model))
    mlflow.set_tag("version.onnx",onnx.__version__)
    mlflow.onnx.log_model(onnx_model, artifact_path, \
        registered_model_name=None if not model_name else f"{model_name}_onnx")
    return onnx_model

def score(model, data_ndarray):
    import onnxruntime as rt
    session = rt.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: data_ndarray.astype(np.float32)})[0]
