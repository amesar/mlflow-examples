# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import platform
import pandas as pd
import numpy as np
import click
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from wine_quality import plot_utils

print("Versions:")
print("  MLflow Version:", mlflow.__version__)
print("  Sklearn Version:", sklearn.__version__)
print("  MLflow Tracking URI:", mlflow.get_tracking_uri())
print("  Python Version:", platform.python_version())
print("  Operating System:", platform.system()+" - "+platform.release())
print("  Platform:", platform.platform())

client = mlflow.tracking.MlflowClient()

colLabel = "quality"

class Trainer():
    def __init__(self, experiment_name, data_path, log_as_onnx, autolog, save_signature, run_origin="none"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        self.log_as_onnx = log_as_onnx
        self.autolog = autolog
        self.save_signature = save_signature
        self.X_train, self.X_test, self.y_train, self.y_test = self.build_data(data_path)

        if self.experiment_name:
            mlflow.set_experiment(experiment_name)

    def build_data(self, data_path):
        data = pd.read_csv(data_path)
        train, test = train_test_split(data, test_size=0.30, random_state=42)
    
        # The predicted column is "quality" which is a scalar from [3, 9]
        X_train = train.drop([colLabel], axis=1)
        X_test = test.drop([colLabel], axis=1)
        y_train = train[[colLabel]]
        y_test = test[[colLabel]]

        return X_train, X_test, y_train, y_test 

    def train(self, max_depth, max_leaf_nodes, model_name, output_path):
        with mlflow.start_run(run_name=self.run_origin) as run:  # NOTE: mlflow CLI ignores run_name
            if self.autolog:
                mlflow.sklearn.autolog()

            run_id = run.info.run_uuid
            experiment_id = run.info.experiment_id
            print("MLflow:")
            print("  run_id:", run_id)
            print("  experiment_id:", experiment_id)
            print("  experiment_name:", client.get_experiment(experiment_id).name)

            # MLflow tags
            mlflow.set_tag("autolog",self.autolog)
            mlflow.set_tag("save_signature",self.save_signature)
            mlflow.set_tag("mlflow.runName", self.run_origin) # mlflow CLI picks this up
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("version.mlflow", mlflow.__version__)
            mlflow.set_tag("version.sklearn", sklearn.__version__)
            mlflow.set_tag("version.platform", platform.platform())
            mlflow.set_tag("version.python", platform.python_version())
            mlflow.set_tag("model_name",model_name)

            # Create model
            dt = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
            print("Model:\n ", dt)

            # Fit and predict
            dt.fit(self.X_train, self.y_train)
            predictions = dt.predict(self.X_test)
            signature = infer_signature(self.X_train, predictions) if self.save_signature else None
            print("signature:",signature)

            # MLflow params
            print("Parameters:")
            print("  max_depth:", max_depth)
            print("  max_leaf_nodes:", max_leaf_nodes)
            if not self.autolog:
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

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
            
                # MLflow log model -  autolog creates a model called "model"
                mlflow.sklearn.log_model(dt, "sklearn-model", registered_model_name=model_name, signature=signature)

            # Convert sklearn model to ONNX and log model
            if self.log_as_onnx:
                from wine_quality import onnx_utils
                onnx_utils.log_model(dt, "onnx-model", model_name, self.X_test)

            # MLflow artifact - plot file
            plot_file = "plot.png"
            plot_utils.create_plot_file(self.y_test, predictions, plot_file)
            mlflow.log_artifact(plot_file)

            # Write run ID to file
            if (output_path):
                mlflow.set_tag("output_path", output_path)
                output_path = output_path.replace("dbfs:","/dbfs")
                with open(output_path, "w") as f:
                    f.write(run_id)

        return (experiment_id,run_id)

@click.command()
@click.option("--experiment_name", help="Experiment name.", default=None, type=str)
@click.option("--data_path", help="Data path.", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--model_name", help="Registered model name.", default=None, type=str)
@click.option("--max_depth", help="Max depth parameter.", default=None, type=int)
@click.option("--max_leaf_nodes", help="Max leaf nodes parameter.", default=32, type=int)
@click.option("--output_path", help="Output file containing run ID.", default="none", type=str)
@click.option("--log_as_onnx", help="Log model as ONNX flavor. Default is false.", default=False, type=bool)
@click.option("--run_origin", help="Run origin.", default="none", type=str)
@click.option("--autolog", help="Autolog parameters and metrics. Default is False.", default=False, type=bool)
@click.option("--save_signature", help="Save model signature. Default is False.", default=False, type=bool)
         
def main(experiment_name, data_path, model_name, max_depth, max_leaf_nodes, log_as_onnx, output_path, autolog, save_signature, run_origin):
    print("Options:")
    for k,v in locals().items(): 
        print(f"  {k}: {v}")
    model_name = None if not model_name or model_name == "None" else model_name
    print("Processed Options:")
    print(f"  model_name: {model_name} - type: {type(model_name)}")
    trainer = Trainer(experiment_name, data_path, log_as_onnx, autolog, save_signature, run_origin)
    trainer.train(max_depth, max_leaf_nodes, model_name, output_path)

if __name__ == "__main__":
    main()
