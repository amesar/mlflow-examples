import mlflow
from wine_quality.train import Trainer
from wine_quality import predict_utils
from wine_quality import onnx_utils

experiment_name = "test_sklearn"
data_path = "../../../data/train/wine-quality-white.csv"
data = predict_utils.read_prediction_data(data_path)

def _test_basic(autolog, model_name):
    trainer = Trainer(experiment_name, data_path, log_as_onnx=False, autolog=autolog, save_signature=False)
    experiment_id,run_id = trainer.train(max_depth=5, max_leaf_nodes=5, model_name=None, output_path=None)
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.sklearn.load_model(model_uri)
    print("model.type:", type(model))

    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    #print("predictions:", predictions)
    assert predictions.shape == (4898,)

    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    assert predictions.shape == (4898,)

def test_basic():
    _test_basic(False, "sklearn-model")

def test_autolog():
    _test_basic(True, "model")

def test_onnx():
    trainer = Trainer(experiment_name, data_path, log_as_onnx=True, autolog=False, save_signature=False)
    _,run_id = trainer.train(max_depth=5, max_leaf_nodes=5, model_name=None, output_path=None)
    model_uri = f"runs:/{run_id}/onnx-model"
    print("model_uri:",model_uri)
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    predictions = onnx_utils.score(model, data.to_numpy())
    predict_utils.display_predictions(predictions)

    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    assert predictions.shape == (4898,1)

    model = mlflow.pyfunc.load_model(model_uri)
    print("model.type:", type(model))
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    assert predictions.shape == (4898,1)

def _test_spark_udf(model_name):
    trainer = Trainer(experiment_name, data_path, log_as_onnx=True, autolog=False, save_signature=False)
    _,run_id = trainer.train(max_depth=5, max_leaf_nodes=5, model_name=None, output_path=None)
    model_uri = f"runs:/{run_id}/{model_name}"
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
    df_data = spark.read.option("inferSchema",True).option("header",True).csv(data_path)
    if "quality" in df_data.columns:
        df_data = df_data.drop("quality")
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = df_data.withColumn("prediction", udf(*df_data.columns))
    print("predictions.type:", type(predictions))
    #predictions.show(10)
    assert predictions.count() == 4898

def test_spark_udf():
    _test_spark_udf("sklearn-model")

def test_spark_udf_onnx():
    _test_spark_udf("onnx-model")
