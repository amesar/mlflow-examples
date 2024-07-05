"""
Trains a wine quality dataset with a sklearn DecisionTreeRegressor using MLflow manual logging.
The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
"""

import platform
import time
import pandas as pd
import numpy as np
import click
import shortuuid

import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from . import plot_utils, mlflow_utils
from . timestamp_utils import fmt_ts_seconds, fmt_ts_millis
from . import common
from . predict import pyfunc_predict


print("Versions:")
print("  MLflow Version:", mlflow.__version__)
print("  Sklearn Version:", sklearn.__version__)
print("  MLflow Tracking URI:", mlflow.get_tracking_uri())
print("  Python Version:", platform.python_version())
print("  Operating System:", platform.system()+" - "+platform.release())
print("  Platform:", platform.platform())

client = mlflow.client.MlflowClient()

col_label = "quality"

now = fmt_ts_seconds(round(time.time()))

class Trainer():
    def __init__(self, experiment_name, data_path, log_as_onnx, log_signature, log_plot=False, run_origin=None):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        self.log_as_onnx = log_as_onnx
        self.log_signature = log_signature
        self.log_plot = log_plot
        self.X_train, self.X_test, self.y_train, self.y_test = self._build_data(data_path)

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)
            exp = client.get_experiment_by_name(experiment_name)
            client.set_experiment_tag(exp.experiment_id, "version_mlflow", mlflow.__version__)
            client.set_experiment_tag(exp.experiment_id, "experiment_created", now)


    def _build_data(self, data_path):
        data = pd.read_csv(data_path)
        data.columns = data.columns.str.replace(" ","_")
        train, test = train_test_split(data, test_size=0.30, random_state=42)

        # The predicted column is "quality" which is a scalar from [3, 9]
        X_train = train.drop([col_label], axis=1)
        X_test = test.drop([col_label], axis=1)
        y_train = train[[col_label]]
        y_test = test[[col_label]]
        return X_train, X_test, y_train, y_test


    def train(self,
            run_name,
            registered_model_name,
            registered_model_version_stage = "None",
            archive_existing_versions = False,
            registered_model_alias = None,
            output_path = None,
            max_depth = None,
            max_leaf_nodes = 32,
            log_input = False,
            log_input_example = False,
            log_evaluation_metrics = False,
            log_shap = False
        ):
        with mlflow.start_run(run_name=run_name) as run: # NOTE: when running with `mlflow run`, mlflow --run-name option takes precedence!
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            print("MLflow run:")
            print("  run_id:", run_id)
            print("  experiment_id:", experiment_id)
            print("  experiment_name:", client.get_experiment(experiment_id).name)

            # MLflow tags
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag("log_signature", self.log_signature)
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("reg_model_name", registered_model_name)
            mlflow.set_tag("reg_model_version_stage", registered_model_version_stage)
            mlflow.set_tag("uuid",shortuuid.uuid())
            mlflow.set_tag("dataset", "wine-quality")
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("timestamp", now)
            mlflow.set_tag("version.mlflow", mlflow.__version__)
            mlflow.set_tag("version.sklearn", sklearn.__version__)
            mlflow.set_tag("version.platform", platform.platform())
            mlflow.set_tag("version.python", platform.python_version())

            # Create model
            model  =  DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
            print("Model:\n ", model)

            # Fit and predict
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            # MLflow params
            print("Parameters:")
            print("  max_depth:", max_depth)
            print("  max_leaf_nodes:", max_leaf_nodes)

            mlflow.log_param("max_depth", max_depth) # NOTE: when running with `mlflow run`, mlflow autologs all -P parameters!!
            mlflow.log_param("max_leaf_nodes", max_leaf_nodes) # ibid

            # MLflow metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            print("Metrics:")
            print("  rmse:", rmse)
            print("  mae:", mae)
            print("  r2:", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # new in MLflow 2.4.0
            if log_input:
                print("Logging input")
                dataset = mlflow.data.from_pandas(self.X_train, source=self.data_path, name="wine_quality_white")
                print("Log input:", dataset)
                mlflow.log_input(dataset, context="training")
                mlflow.set_tag("log_input", True)

            # Create signature
            signature = infer_signature(self.X_train, predictions) if self.log_signature else None
            print("Signature:",signature)

            # Input example
            input_example = self.X_test.head(5) if log_input_example else None

            # Log MLflow model
            mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

            # Add evaluation metrics to run with mlflow.evaluate
            if log_evaluation_metrics:
                model_uri = mlflow.get_artifact_uri("model")
                print("Evaluation model_uri:", model_uri)
                data = pd.concat([self.X_test, self.y_test], axis=1)
                feature_columns = list(self.X_test.columns)
                results = mlflow.evaluate(
                    model = model_uri,
                    data = data,
                    targets = "quality",
                    model_type = "regressor",
                    evaluators = "default",
                    feature_names = feature_columns,
                    evaluator_config = {"explainability_nsamples": 1000},
                )
                mlflow_utils.log_dict(results.metrics, "evaluation_metrics.json")

            if registered_model_name:
                mlflow_utils.register_model(run,
                    "model",
                    registered_model_name,
                    registered_model_version_stage,
                    archive_existing_versions,
                    registered_model_alias,
                    run_name
                )

            # Convert sklearn model to ONNX and log model
            if self.log_as_onnx:
                from wine_quality import onnx_utils
                onnx_utils.log_model(model, "onnx_model", self.X_test, signature, input_example)
                if registered_model_name:
                    if registered_model_alias:
                        registered_model_alias = f"{registered_model_alias}_onnx"
                    mlflow_utils.register_model(run,
                        "onnx_model",
                        f"{registered_model_name}_onnx",
                        registered_model_version_stage,
                        archive_existing_versions,
                        registered_model_alias,
                        run_name
                    )

            # MLflow artifact - plot file
            if self.log_plot:
                plot_file = "plot.png"
                plot_utils.create_plot_file(self.y_test, predictions, plot_file)
                mlflow.log_artifact(plot_file)

            # Write run ID to file
            if (output_path):
                mlflow.set_tag("output_path", output_path)
                output_path = output_path.replace("dbfs:","/dbfs")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(run_id)
            if log_shap:
                mlflow.shap.log_explanation(model.predict, self.X_train, "shap") # TODO: loops forever

        run = client.get_run(run_id)
        client.set_tag(run_id, "run.info.start_time", run.info.start_time)
        client.set_tag(run_id, "run.info.end_time", run.info.end_time)
        client.set_tag(run_id, "run.info._start_time", fmt_ts_millis(run.info.start_time))
        client.set_tag(run_id, "run.info._end_time", fmt_ts_millis(run.info.end_time))

        return run_id


@click.command()
@click.option("--experiment-name",
    help="Experiment name.",
    type=str,
    default=None,
    show_default=True
)
@click.option("--run-name",
    help="Run name",
    type=str,
    required=False
)
@click.option("--data-path",
    help="Data path.",
    type=str,
    default=common.data_path,
    show_default=True
)
@click.option("--model-name",
    help="Registered model name.",
    type=str,
    default=None,
    show_default=True
)
@click.option("--model-version-stage",
    help="Registered model version stage: production|staging|archive|none.",
    type=str,
    default=None,
    show_default=True
)
@click.option("--archive-existing-versions",
    help="Archive existing versions.",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--model-alias",
    help="Registered model alias",
    type=str,
    required=False
)
@click.option("--log-signature",
    help="Log model signature.",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--log-as-onnx",
    help="Log model as ONNX flavor.",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--max-depth",
    help="Max depth parameter.",
    type=int,
    default=None,
    show_default=True
)
@click.option("--max-leaf-nodes",
    help="Max leaf nodes parameter.",
    type=int,
    default=32,
    show_default=True
)
@click.option("--run-origin",
    help="Run origin.",
    type=str,
    default="none",
    show_default=True
)
@click.option("--log-input",
    help="Log data input (automatically creates signature)",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--log-input-example",
    help="Log input example (automatically creates signature)",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--log-evaluation-metrics",
    help="Log metrics from mlflow.evaluate",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--log-shap",
    help="Log mlflow.shap.log_explanation",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--log-plot",
    help="Log plot",
    type=bool,
    default=False,
    show_default=True
)
@click.option("--output-path",
    help="Output file containing run ID.",
    type=str,
    default=None,
    show_default=True
)
def main(experiment_name,
        run_name,
        data_path,
        model_name,
        model_version_stage,
        archive_existing_versions,
        model_alias,
        log_signature,
        log_as_onnx,
        max_depth,
        max_leaf_nodes,
        run_origin,
        log_input,
        log_input_example,
        log_evaluation_metrics,
        log_shap,
        log_plot,
        output_path
    ):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    print("Processed Options:")
    print(f"  model_name: {model_name} - type: {type(model_name)}")

    # Train
    trainer = Trainer(experiment_name, data_path, log_as_onnx, log_signature, log_plot, run_origin)
    run_id = trainer.train(run_name, model_name, model_version_stage,
        archive_existing_versions, model_alias, output_path,
        max_depth, max_leaf_nodes, log_input, log_input_example, log_evaluation_metrics, log_shap
    )

    # Predict
    model_uri = f"runs:/{run_id}/model"
    pyfunc_predict(model_uri, data_path)

    print(f"\nTraining run succeded with run ID '{run_id}.'\n")


if __name__ == "__main__":
    main()
