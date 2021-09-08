import pytest
import mlflow
from wine_quality.autolog_train import Trainer
from wine_quality import predict

data_path = "../../../data/train/wine-quality-white.csv"
run_id = None
experiment_name = "sklearn_test"
client = mlflow.tracking.MlflowClient()

def sklearn_model_uri():  
    return f"runs:/{run_id}/model"

def banner(msg, model_uri):
    print("\n\n********************")
    print(f"** Test: {msg} {model_uri}")
    print("********************")

def get_last_run_id():
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(exp.experiment_id, order_by=["attribute.end_time DESC"], max_results=1)
    return runs[0].info.run_id

@pytest.mark.order1
def test_train():
    global run_id
    banner("test_train","")
    trainer = Trainer(experiment_name, data_path)
    trainer.train(5, 5)
    run_id = get_last_run_id()
    print("run_id:",run_id)


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
