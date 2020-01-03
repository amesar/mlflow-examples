import onnx

def log_model(spark, model, name, data_df):
    import mlflow.onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    initial_types = [ (col, FloatTensorType([1, 1])) for col in data_df.columns ]
    onnx_model = onnxmltools.convert_sparkml(model, name, initial_types, spark_session=spark)
    mlflow.onnx.log_model(onnx_model, "onnx-model")
    mlflow.set_tag("onnx_version", onnx.__version__)

def score_model(model, data_ndarray):
    import numpy as np
    import onnxruntime
    sess = onnxruntime.InferenceSession(model.SerializeToString())
    input_feed = { col.name: data_ndarray.astype(np.float32) for col in sess.get_inputs() }
    return sess.run(None, input_feed)[0]

# NOTE: for ONNX bug/feature - columns with spaces
def normalize_columns(columns):
    return [ col.replace(" ","_") for col in columns ]
