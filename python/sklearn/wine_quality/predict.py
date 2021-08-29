import click
import mlflow
import mlflow.sklearn
from wine_quality import predict_utils

def banner(msg, model_uri):
    print("\n==========")
    print(f"Predict: {msg} - {model_uri}")
    print("==========\n")

def _predict(model_uri, data_path, load_model_method, msg):
    banner(msg, model_uri)
    print("data_path:", data_path)
    print("model_uri:", model_uri)

    model = load_model_method(model_uri)
    print("model.type:", type(model))

    data = predict_utils.read_prediction_data(data_path)
    predictions = model.predict(data)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    print("predictions:", predictions)

def sklearn_predict(model_uri, data_path):
    _predict(model_uri, data_path, mlflow.sklearn.load_model, "sklearn")

def pyfunc_predict(model_uri, data_path):
    _predict(model_uri, data_path, mlflow.pyfunc.load_model, "sklearn")

def spark_udf_predict(model_uri, data_path):
    banner("spark_udf", model_uri)
    from pyspark.sql import SparkSession
    print("data_path:", data_path)
    print("model_uri:", model_uri)
    print("MLflow Version:", mlflow.__version__)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
    print("Spark Version:", spark.version)

    data = spark.read.option("inferSchema",True).option("header",True).csv(data_path) if data_path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(data_path)
    if "quality" in data.columns:
        data = data.drop("quality")

    print("\nUDF with DataFrame API")
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    predictions = data.withColumn("prediction", udf(*data.columns))
    predictions.show(10)

    print("UDF with SQL")
    spark.udf.register("predictUDF", udf)
    data.createOrReplaceGlobalTempView("data")
    predictions = spark.sql("select *, predictUDF(*) as prediction from global_temp.data")
    predictions.show(10)

def onnx_predict(model_uri, data_path):
    _predict(model_uri, data_path, mlflow.pyfunc.load_model, "onnx_predict")

predict_methods = {
    "sklearn": sklearn_predict, 
    "pyfunc": pyfunc_predict, 
    "spark_udf": spark_udf_predict, 
    "onnx": onnx_predict,
    "onnx_pyfunc": pyfunc_predict
}

@click.command()
@click.option("--model-uri", help="Model URI.", required=True, type=str)
@click.option("--data-path", help="Data path.", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--flavor", help="MLflow flavor.", required=True, type=str)
def main(model_uri, data_path, flavor):
    print("Options:")
    for k,v in locals().items(): print(f"  {k}: {v}")
    method = predict_methods.get(flavor, None)
    if not method:
        print(f"ERROR: Unknown flavor '{flavor}'. Must be one of: {set(predict_methods.keys())}.")
    else:
        method(model_uri, data_path)

if __name__ == "__main__":
    main()
