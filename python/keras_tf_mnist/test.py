import train
import keras_predict
import pyfunc_predict
from click.testing import CliRunner

def now():
    import time
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime(time.time()))

id = now()
experiment_name = f"test_keras_mnist_{id}"
model_name = f"test_keras_mnist_{id}"
args = [ "--experiment_name", experiment_name, "--model_name", model_name]
autolog_args = args + [ "--mlflow_custom_log", False] 

runner = CliRunner()

def test_non_autolog():
    res = runner.invoke(train.main, args)
    assert_res(res,"train")
    run_id = get_run_id(res.output)
    predict(f"runs:/{run_id}/keras-model")
    predict(f"models:/{model_name}/1")

def test_mlflow_autolog():
    res = runner.invoke(train.main, autolog_args + [ "--mlflow_autolog", True])
    assert_res(res,"train")
    run_id = get_run_id(res.output)
    predict(f"runs:/{run_id}/model")
    predict(f"models:/{model_name}/1")

def test_tensorflow_autolog():
    res = runner.invoke(train.main, autolog_args + [ "--tensorflow_autolog", True])
    assert_res(res,"train")
    run_id = get_run_id(res.output)
    predict(f"runs:/{run_id}/model")
    predict(f"models:/{model_name}/1")

def test_keras_autolog(): # Expected failure inside MLflow
    res = runner.invoke(train.main, autolog_args + [ "--keras_autolog", True])
    print(">> keras_autolog.res:",res)
    assert res.exit_code == 1
    assert """ModuleNotFoundError("No module named 'keras'")""" in str(res)

def predict(model_uri):
    print(">> model_uri:",model_uri)
    res = runner.invoke(pyfunc_predict.main, [ "--model_uri", model_uri])
    assert_res(res,"pyfunc_predict")
    res = runner.invoke(keras_predict.main, [ "--model_uri", model_uri])
    assert_res(res,"keras_predict")

def get_run_id(output):
    toks = output.split(" ")
    idx = toks.index("run_id:")
    return toks[idx+1].strip()

def assert_res(res, msg):
    print(f">> {msg}.res:",res)
    assert res.exit_code == 0
