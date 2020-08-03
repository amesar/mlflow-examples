
def log_model(model, artifact_path, model_name=None):
    import mlflow
    import mlflow.onnx
    import onnx
    import onnxmltools
    onnx_model = onnxmltools.convert_keras(model, artifact_path)
    print("onnx_model.type:",type(onnx_model))
    mlflow.onnx.log_model(onnx_model, artifact_path, registered_model_name=model_name)
    mlflow.set_tag("version.onnx",onnx.__version__)
    mlflow.set_tag("version.onnxtools",onnxmltools.__version__)

def score_model(model, data):
    import numpy as np
    import onnxruntime 
    session = onnxruntime.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: data.astype(np.float32)})[0]
