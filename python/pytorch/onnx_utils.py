
def log_model(model, artifact_path, data):
    import io
    import torch
    import onnx
    import mlflow.onnx
    mlflow.set_tag("onnx_version",onnx.__version__)
    f = io.BytesIO()
    torch.onnx.export(model, data, f)
    onnx_model = f.getvalue()
    mlflow.onnx.log_model(onnx_model, artifact_path)

def score_model(model, data):
    import numpy as np
    import onnxruntime 
    session = onnxruntime.InferenceSession(model.SerializeToString())
    input_name = session.get_inputs()[0].name
    print("data.type:",type(data))
    return session.run(None, {input_name: data.astype(np.float32)})[0]
