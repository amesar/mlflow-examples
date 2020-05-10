import onnx

def log_model(spark, model, name, model_name, data_df):
    import mlflow.onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert.sparkml.utils import buildInitialTypesSimple
    initial_types = buildInitialTypesSimple(data_df)
    onnx_model = onnxmltools.convert_sparkml(model, name, initial_types, spark_session=spark)
    mlflow.onnx.log_model(onnx_model, "onnx-model", \
        registered_model_name=None if not model_name else f"{model_name}_onnx")
    mlflow.set_tag("version.onnx", onnx.__version__)
    mlflow.set_tag("version.onnxmltools", onnxmltools.__version__)

def score_model(model, data_ndarray):
    import numpy as np
    import onnxruntime
    session = onnxruntime.InferenceSession(model.SerializeToString())
    input_feed = { col.name: data_ndarray[:,j].astype(np.float32) for j,col in enumerate(session.get_inputs()) }
    return session.run(None, input_feed)[0]


# NOTE: for ONNX bug/feature - columns with spaces
def normalize_columns(columns):
    return [ col.replace(" ","_") for col in columns ]
