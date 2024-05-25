import pytest
from wine_quality.train import Trainer
from wine_quality import predict
from . common import create_run, banner, sklearn_model_uri, onnx_model_uri, data_path

@pytest.mark.order1
def test_train():
    create_run()

@pytest.mark.order2
def test_sklearn_predict():
    banner("test_sklearn_predict",sklearn_model_uri())
    predict.sklearn_predict(sklearn_model_uri(), data_path)

@pytest.mark.order3
def test_pyfunc_predict():
    banner("test_pyfunc_predict",sklearn_model_uri())
    predict.pyfunc_predict(sklearn_model_uri(), data_path)

@pytest.mark.order5
def test_onnx_predict():
    banner("test_onnx_predict",onnx_model_uri())
    predict.onnx_predict(onnx_model_uri(), data_path)

@pytest.mark.order6
def test_onnx_pyfunc_predict():
    banner("test_onnx_pyfunc_predict",onnx_model_uri())
    predict.pyfunc_predict(onnx_model_uri(), data_path)
