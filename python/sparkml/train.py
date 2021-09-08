"""
PySpark Decision Tree Regression example.
"""

import platform
import click
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow
import mlflow.spark
import common
from sparkml_udf_workaround import log_udf_model

spark = SparkSession.builder.appName("App").getOrCreate()
common.show_versions(spark)

def train(run_id, data, max_depth, max_bins, model_name, log_as_mleap, log_as_onnx, spark_autolog):
    (trainingData, testData) = data.randomSplit([0.7, 0.3], 42)
    print("testData.schema:")
    testData.printSchema()

    # MLflow - log parameters
    print("Parameters:")
    print("  max_depth:", max_depth)
    print("  max_bins:", max_bins)
    if not spark_autolog:
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_bins", max_bins)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=common.colLabel, featuresCol=common.colFeatures, maxDepth=max_depth, maxBins=max_bins)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=common.colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model and predict
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # MLflow - log metrics
    print("Metrics:")
    predictions = model.transform(testData)
    metrics = ["rmse", "r2", "mae"]
    for metric_name in metrics:
        evaluator = RegressionEvaluator(labelCol=common.colLabel, predictionCol=common.colPrediction, metricName=metric_name)
        metric_value = evaluator.evaluate(predictions)
        print(f"  {metric_name}: {metric_value}")
        if not spark_autolog:
            mlflow.log_metric(metric_name,metric_value)

    # MLflow - log spark model
    if not spark_autolog:
        mlflow.spark.log_model(model, "spark-model", registered_model_name=model_name)

        # MLflow - log Spark model with UDF wrapper for workaround
        log_udf_model(run_id, "spark-model", data.columns, model_name)

    # MLflow - log as MLeap model
    if log_as_mleap:
        scoreData = testData.drop("quality")
        mlflow.mleap.log_model(spark_model=model, sample_input=scoreData, artifact_path="mleap-model", \
            registered_model_name=None if not model_name else f"{model_name}_mleap")

        # Log MLeap schema file for MLeap runtime deserialization
        schema_path = "schema.json"
        with open(schema_path, 'w') as f:
            f.write(scoreData.schema.json())
        print("schema_path:", schema_path)
        mlflow.log_artifact(schema_path, "mleap-model")

    # MLflow - log as ONNX model
    if log_as_onnx:
        import onnx_utils
        scoreData = testData.drop("quality")
        onnx_utils.log_model(spark, model, "onnx-model", model_name, scoreData)

@click.command()
@click.option("--experiment-name", help="Experiment name", default=None, type=str)
@click.option("--data-path", help="Data path", default=common.default_data_path, type=str)
@click.option("--model-name", help="Registered model name", default=None, type=str)
@click.option("--max-depth", help="Max depth", default=5, type=int) # per doc
@click.option("--max-bins", help="Max bins", default=32, type=int) # per doc
@click.option("--describe", help="Describe data", default=False, type=bool)
@click.option("--log-as-mleap", help="Score as MLeap", default=False, type=bool)
@click.option("--log-as-onnx", help="Log model as ONNX flavor", default=False, type=bool)
@click.option("--spark-autolog", help="Use spark.autolog", default=False, type=bool)

def main(experiment_name, model_name, data_path, max_depth, max_bins, describe, log_as_mleap, log_as_onnx, spark_autolog):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")

    client = mlflow.tracking.MlflowClient()
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    if spark_autolog:
        SparkSession.builder.config("spark.jars.packages", "org.mlflow.mlflow-spark")
        mlflow.spark.autolog()
    data_path = data_path or common.default_data_path
    data = common.read_data(spark, data_path)
    if (describe):
        print("==== Data")
        data.describe().show()

    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        print("  experiment_name:", client.get_experiment(run.info.experiment_id).name)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
        mlflow.set_tag("version.spark", spark.version)
        mlflow.set_tag("version.pyspark", pyspark.__version__)
        mlflow.set_tag("version.os", platform.system()+" - "+platform.release())
        model_name = None if model_name is None or model_name == "None" else model_name
        train(run.info.run_id, data, max_depth, max_bins, model_name, log_as_mleap, log_as_onnx, spark_autolog)

if __name__ == "__main__":
    main()
