"""
PySpark Decision Tree Regression Example.
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import mlflow
import mlflow.spark
from common import *

spark = SparkSession.builder.appName("App").getOrCreate()

print("MLflow Version:", mlflow.version.VERSION)
print("Spark Version:", spark.version)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

metrics = ["rmse", "r2", "mae"]

def train(data, maxDepth, maxBins, run_id, log_as_onnx):
    (trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)
    print("testData.schema:")
    testData.printSchema()

    # MLflow - log parameters
    print("Parameters:")
    print("  maxDepth:", maxDepth)
    print("  maxBins:", maxBins)
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("maxBins", maxBins)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, maxDepth=maxDepth, maxBins=maxBins)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model and predict
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # MLflow - log metrics
    print("Metrics:")
    predictions = model.transform(testData)
    for metric_name in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric_name)
        metric_value = evaluator.evaluate(predictions)
        print(f"  {metric_name}: {metric_value}")
        mlflow.log_metric(metric_name,metric_value)

    # MLflow - log spark model
    mlflow.spark.log_model(model, "spark-model")

    # MLflow - log mleap model
    mleapData = testData.drop("quality")
    mlflow.mleap.log_model(spark_model=model, sample_input=mleapData, artifact_path="mleap-model")

    # Log mleap schema file for MLeap runtime deserialization
    schema_path = "schema.json"
    with open(schema_path, 'w') as f:
        f.write(mleapData.schema.json())
    print("schema_path:", schema_path)
    mlflow.log_artifact(schema_path, "mleap-model")

    # MLflow - log onnx model
    if log_as_onnx:
        import onnx_utils
        onnx_utils.log_model(spark, model, "onnx-model", mleapData)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment _name", default="pyspark")
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=default_data_path)
    parser.add_argument("--max_depth", dest="max_depth", help="Max depth", default=5, type=int) # per doc
    parser.add_argument("--max_bins", dest="max_bins", help="Max bins", default=32, type=int) # per doc
    parser.add_argument("--describe", dest="describe", help="Describe data", default=False, action='store_true')
    parser.add_argument("--log_as_onnx", dest="log_as_onnx", help="Log model as ONNX", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment(args.experiment_name)

    data_path = args.data_path or default_data_path
    data = read_data(spark, data_path)
    if (args.describe):
        print("==== Data")
        data.describe().show()

    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:", run.info.run_id)
        print("  experiment_id:", run.info.experiment_id)
        print("  experiment_name:", client.get_experiment(run.info.experiment_id).name)
        mlflow.set_tag("mlflow_version", mlflow.version.VERSION)
        mlflow.set_tag("spark_version", spark.version)
        train(data, args.max_depth, args.max_bins, run.info.run_id, args.log_as_onnx)
