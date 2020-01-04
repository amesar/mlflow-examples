from pyspark.ml import PipelineModel
from mleap.pyspark.spark_support import SimpleSparkSerializer

"""
- Unfortunately MLflow does not have an API method to read in a model as a Spark Bundle.
- So we have to manually construct the bundle URI and directly deserialize it with MLeap methods.
"""
def load_model(run, artifact_path):
    bundle_uri = f"{run.info.artifact_uri}/{artifact_path}"
    print("bundle_uri:", bundle_uri)
    return PipelineModel.deserializeFromBundle(bundle_uri)
