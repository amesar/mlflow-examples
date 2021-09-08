"""
Trains a wine quality dataset with a sklearn DecisionTreeRegressor using MLflow autologging.
"""

import platform
import pandas as pd
#import numpy as np
import click
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn

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
    def __init__(self, experiment_name, data_path):
        self.experiment_name = experiment_name
        self.data_path = data_path
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

    def train(self, max_depth, max_leaf_nodes):
        mlflow.sklearn.autolog()
        model = DecisionTreeRegressor(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        print("Model:\n ", model)
        model.fit(self.X_train, self.y_train)
        #predictions = model.predict(self.X_test)

@click.command()
@click.option("--experiment-name", help="Experiment name.", default=None, type=str)
@click.option("--data-path", help="Data path.", default="../../data/train/wine-quality-white.csv", type=str)
@click.option("--max-depth", help="Max depth parameter.", default=None, type=int)
@click.option("--max-leaf-nodes", help="Max leaf nodes parameter.", default=32, type=int)
         
def main(experiment_name, data_path, max_depth, max_leaf_nodes):
    print("Options:")
    for k,v in locals().items(): 
        print(f"  {k}: {v}")
    print("Processed Options:")
    trainer = Trainer(experiment_name, data_path)
    trainer.train(max_depth, max_leaf_nodes)

if __name__ == "__main__":
    main()
