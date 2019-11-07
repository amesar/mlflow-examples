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
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

metrics = ["rmse","r2", "mae"]

def train(data, maxDepth, maxBins, run_id):
    (trainingData, testData) = data.randomSplit([0.7, 0.3], 2019)

    # MLflow - log parameters
    print("Parameters:")
    print("  maxDepth:",maxDepth)
    print("  maxBins:",maxBins)
    mlflow.log_param("maxDepth",maxDepth)
    mlflow.log_param("maxBins",maxBins)

    # Create pipeline
    dt = DecisionTreeRegressor(labelCol=colLabel, featuresCol=colFeatures, maxDepth=maxDepth, maxBins=maxBins)
    assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol=colFeatures)
    pipeline = Pipeline(stages=[assembler, dt])
    
    # Fit model and predic
    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    # MLflow - log metrics
    print("Metrics:")
    predictions = model.transform(testData)
    for metric_name in metrics:
        evaluator = RegressionEvaluator(labelCol=colLabel, predictionCol=colPrediction, metricName=metric_name)
        metric_value = evaluator.evaluate(predictions)
        print("  {}: {}".format(metric_name,metric_value))
        mlflow.log_metric(metric_name,metric_value)

    # MLflow - log model
    mlflow.spark.log_model(model, "spark-model")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", dest="experiment_name", help="experiment_name", default="pyspark")
    parser.add_argument("--data_path", dest="data_path", help="data_path", default=default_data_path)
    parser.add_argument("--max_depth", dest="max_depth", help="max_depth", default=2, type=int)
    parser.add_argument("--max_bins", dest="max_bins", help="max_bins", default=32, type=int)
    parser.add_argument("--describe", dest="describe", help="Describe data", default=False, action='store_true')
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print("  {}: {}".format(arg,getattr(args, arg)))

    client = mlflow.tracking.MlflowClient()
    print("experiment_name:",args.experiment_name)
    mlflow.set_experiment(args.experiment_name)
    print("experiment_id:",client.get_experiment_by_name(args.experiment_name).experiment_id)

    data_path = args.data_path or default_data_path
    data = read_data(spark, data_path)
    if (args.describe):
        print("==== Data")
        data.describe().show()

    with mlflow.start_run() as run:
        print("MLflow:")
        print("  run_id:",run.info.run_id)
        print("  experiment_id:",run.info.experiment_id)
        train(data, args.max_depth, args.max_bins, run.info.run_id)
