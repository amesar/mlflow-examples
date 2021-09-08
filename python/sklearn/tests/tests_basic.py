import pytest
from wine_quality.train import Trainer
from wine_quality import predict

data_path = "../../../data/train/wine-quality-white.csv"
run_id = None
experiment_name = "sklearn_test"

def sklearn_model_uri():  
    return f"runs:/{run_id}/model"

def onnx_model_uri():  
    return f"runs:/{run_id}/onnx-model"

def banner(msg, model_uri):
    print("\n\n********************")
    print(f"** Test: {msg} {model_uri}")
    print("********************")

@pytest.mark.order1
def test_train():
    global run_id
    banner("test_train","")
    trainer = Trainer(experiment_name, log_as_onnx=True, run_origin="test", data_path=data_path, save_signature=True)
    _, run_id = trainer.train(5, 5, None, "none")


@pytest.mark.order2
def test_sklearn_predict():
    banner("test_sklearn_predict",sklearn_model_uri())
    predict.sklearn_predict(sklearn_model_uri(), data_path)

@pytest.mark.order3
def test_pyfunc_predict():
    banner("test_pyfunc_predict",sklearn_model_uri())
    predict.pyfunc_predict(sklearn_model_uri(), data_path)

@pytest.mark.order4
def test_spark_udf_predict():
    banner("test_spark_udf_predict",sklearn_model_uri())
    predict.spark_udf_predict(sklearn_model_uri(), data_path)

@pytest.mark.order5
def test_onnx_predict():
    banner("test_onnx_predict",onnx_model_uri())
    predict.onnx_predict(onnx_model_uri(), data_path)

@pytest.mark.order6
def test_onnx_pyfunc_predict():
    banner("test_onnx_pyfunc_predict",onnx_model_uri())
    predict.pyfunc_predict(onnx_model_uri(), data_path)

@pytest.mark.order7
def test_onnx_spark_udf_predict():
    banner("test_spark_udf_predict",onnx_model_uri())
    predict.spark_udf_predict(onnx_model_uri(), data_path)

