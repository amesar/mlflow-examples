import numpy as np
import mlflow
import mlflow.onnx


#def log_model(model, mlflow_model_name, registered_model_name, data):
def log_model(model, mlflow_model_name, data, signature=None, input_example=None):
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print("ONNX mlflow_model_name:", mlflow_model_name)
    initial_types = [('float_input', FloatTensorType([None, data.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types)
    print("ONNX model type:", type(onnx_model))
    mlflow.set_tag("version.onnx", onnx.__version__)
    print("ONNX version:", onnx.__version__)

    mlflow.onnx.log_model(onnx_model, mlflow_model_name, signature=signature, input_example=input_example)
    return onnx_model


def score(model, data_ndarray):
    import onnxruntime as rt
    session = rt.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: data_ndarray.astype(np.float32)})[0]
