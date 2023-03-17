import sys
import pandas as pd
import click
import mlflow
import mlflow.sklearn
from wine_quality import predict_utils
from wine_quality import common
from wine_quality import onnx_utils


def banner(msg, model_uri):
    print("\n+===============================================================")
    print(f"|")
    print(f"| {msg} - {model_uri}")
    print(f"|")
    print("+===============================================================\n")


def _predict(model_uri, data_path, load_model_method, msg):
    banner(msg, model_uri)
    print("msg:", msg)
    print("model_uri:", model_uri)
    print("data_path:", data_path)

    model = load_model_method(model_uri)
    print("model.type:", type(model))

    df = predict_utils.read_prediction_data(data_path)
    predictions = model.predict(df)
    print("predictions.type:", type(predictions))
    print("predictions.shape:", predictions.shape)
    if isinstance(predictions, pd.DataFrame):
        print(predictions.head(10).to_string(index=False, justify="right"))
    else:
        print("predictions:", predictions)


def sklearn_predict(model_uri, data_path):
    _predict(model_uri, data_path, mlflow.sklearn.load_model,sys._getframe().f_code.co_name)


def pyfunc_predict(model_uri, data_path):
    _predict(model_uri, data_path, mlflow.pyfunc.load_model, sys._getframe().f_code.co_name)


def onnx_predict(model_uri, data_path):
    banner(sys._getframe().f_code.co_name, model_uri)
    model = mlflow.onnx.load_model(model_uri)
    print("model.type:", type(model))
    df = predict_utils.read_prediction_data(data_path)
    predictions = onnx_utils.score(model, df.to_numpy())
    print("predictions.type:", type(predictions))
    print("predictions:", predictions)


def spark_udf_predict(model_uri, data_path):
    banner("spark_udf", model_uri)
    from pyspark.sql import SparkSession
    import pyspark
    print("pyspark version:",pyspark.__version__)
    print("data_path:", data_path)
    print("model_uri:", model_uri)
    print("MLflow Version:", mlflow.__version__)
    print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    spark = SparkSession.builder.appName("ServePredictions").getOrCreate()
    print("Spark Version:", spark.version)

    df = spark.read.option("inferSchema",True).option("header",True).csv(data_path) if data_path.endswith(".csv") \
    else spark.read.option("multiLine",True).json(data_path)
    if "quality" in df.columns:
        df = df.drop("quality")

    print("\nUDF with DataFrame API")
    udf = mlflow.pyfunc.spark_udf(spark, model_uri)
    print("model.type:", type(udf))
    predictions = df.withColumn("prediction", udf(*df.columns))
    predictions.show(10)

    print("UDF with SQL")
    spark.udf.register("predictUDF", udf)
    df.createOrReplaceGlobalTempView("data")
    predictions = spark.sql("select *, predictUDF(*) as prediction from global_temp.data")
    predictions.show(10)


predict_methods = {
    "sklearn": sklearn_predict, 
    "pyfunc": pyfunc_predict, 
    "onnx": onnx_predict, 
    "spark_udf": spark_udf_predict,
}


@click.command()
@click.option("--model-uri", help="Model URI.", required=True, type=str)
@click.option("--data-path", help="Data path.", default=common.data_path)
@click.option("--flavor", help="MLflow flavor.", required=True, type=str)
def main(model_uri, data_path, flavor):
    print("Options:")
    for k,v in locals().items(): 
        print(f"  {k}: {v}")
    method = predict_methods.get(flavor, None)
    print("method:",method.__name__)
    if not method:
        print(f"ERROR: Unknown flavor '{flavor}'. Must be one of: {set(predict_methods.keys())}.")
    else:
        method(model_uri, data_path)

if __name__ == "__main__":
    main()
