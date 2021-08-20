"""
Workaround for MLflow Spark UDF named columns bug.
"""
import mlflow.pyfunc
import mlflow.spark

class UdfModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ordered_df_columns, model_artifact):
        self.ordered_df_columns = ordered_df_columns
        self.model_artifact = model_artifact

    def load_context(self, context):
        import mlflow.pyfunc
        self.spark_pyfunc = mlflow.pyfunc.load_model(context.artifacts[self.model_artifact])

    def predict(self, context, model_input):
        renamed_input = model_input.rename(
            columns={
                str(index): column_name for index, column_name
                    in list(enumerate(self.ordered_df_columns))
            }
        )
        return self.spark_pyfunc.predict(renamed_input)

def log_udf_model(run_id, artifact_path, ordered_columns, model_name=None):
    udf_artifact_path = f"udf-{artifact_path}"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.pyfunc.log_model(
        artifact_path = udf_artifact_path,
        python_model = UdfModelWrapper(ordered_columns, artifact_path),
        artifacts={ artifact_path: model_uri }, 
        registered_model_name=None if not model_name else f"udf_{model_name}"
    )
    return udf_artifact_path

def log_spark_and_udf_models(model, artifact_path, run_id, ordered_columns):   
  mlflow.spark.log_model(model, artifact_path)
  return log_udf_model(artifact_path, ordered_columns, run_id)
