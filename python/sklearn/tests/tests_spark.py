import pytest
from wine_quality.train import Trainer
from wine_quality import predict
from . common import create_run, banner, sklearn_model_uri, onnx_model_uri, data_path


@pytest.mark.order1
def test_train():
    create_run()


@pytest.mark.order4
def test_spark_udf_predict():
    banner("test_spark_udf_predict",sklearn_model_uri())
    predict.spark_udf_predict(sklearn_model_uri(), data_path)

@pytest.mark.order7
def _test_onnx_spark_udf_predict():
    banner("test_spark_udf_predict",onnx_model_uri())
    predict.spark_udf_predict(onnx_model_uri(), data_path)
