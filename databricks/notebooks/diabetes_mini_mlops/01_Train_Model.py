# Databricks notebook source
# MAGIC %md # MLflow quickstart: training and logging  
# MAGIC
# MAGIC This tutorial is based on the MLflow [ElasticNet Diabetes example](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_diabetes). It illustrates how to use MLflow to track the model training process, including logging model parameters, metrics, the model itself, and other artifacts like plots. It also includes instructions for viewing the logged results in the MLflow tracking UI.    
# MAGIC
# MAGIC This notebook uses the scikit-learn `diabetes` dataset and predicts the progression metric (a quantitative measure of disease progression after one year) based on BMI, blood pressure, and other measurements. It uses the scikit-learn ElasticNet linear regression model, varying the `alpha` and `l1_ratio` parameters for tuning. For more information on ElasticNet, refer to:
# MAGIC   * [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
# MAGIC   * [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/TALKS/enet_talk.pdf)
# MAGIC   * [sklearn.datasets.load_diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)
# MAGIC   
# MAGIC ### Requirements
# MAGIC * This notebook requires Databricks Runtime ML 13.x or above.
# MAGIC

# COMMAND ----------

# MAGIC %md ### Setup

# COMMAND ----------

# MAGIC %run ./includes/Common

# COMMAND ----------

dbutils.widgets.text("Table", "")
table_name = dbutils.widgets.get("Table")
table_name = table_name or None

print("table_name:", table_name)

# COMMAND ----------

# MAGIC %md ### Clean experiment by deleting existing runs

# COMMAND ----------

experiment = get_experiment()

# COMMAND ----------

delete_runs(experiment)

# COMMAND ----------

# MAGIC %md ### Import libraries 

# COMMAND ----------

import os
import warnings

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md ### Load Diabetes dataset

# COMMAND ----------

data = load_data(table_name)

# COMMAND ----------

X = data.drop(["progression"], axis=1).to_numpy()

y = data[["progression"]]
y = np.concatenate(y.to_numpy()).ravel()

# COMMAND ----------

display(data)

# COMMAND ----------

# MAGIC %md #### New

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ### Create function to plot ElasticNet descent path
# MAGIC
# MAGIC The `plot_enet_descent_path()` function:
# MAGIC * Creates and saves a plot of the [ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html) for the ElasticNet model for the specified *l1_ratio*.
# MAGIC * Returns an image that can be displayed in the notebook using `display()`
# MAGIC * Saves the figure `ElasticNet-paths.png` to the cluster driver node

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image
    
    print("Computing regularization path using ElasticNet")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio)

    # Display results
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    image = fig

    fig.savefig("ElasticNet-paths.png")
    plt.close(fig)

    return image    

# COMMAND ----------

# MAGIC %md ### Train the diabetes model
# MAGIC The `train_diabetes()` function trains ElasticNet linear regression based on the input parameters *in_alpha* and *in_l1_ratio*.
# MAGIC
# MAGIC The function uses MLflow Tracking to record the following:
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * the image created by the `plot_enet_descent_path()` function defined previously.
# MAGIC
# MAGIC **Tip:** Databricks recommends using `with mlflow.start_run:` to create a new MLflow run. The `with` context closes the MLflow run regardless of whether the code completes successfully or exits with an error, and you do not have to call `mlflow.end_run`.

# COMMAND ----------

# train_diabetes
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

def train_diabetes(data, alpha, l1_ratio):
  
    # Evaluate metrics
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
    train_x = train.drop(["progression"], axis=1)
    test_x = test.drop(["progression"], axis=1)
    train_y = train[["progression"]]
    test_y = test[["progression"]]

    run_name = f"alpha={alpha} - l1_ratio={l1_ratio}"
    with mlflow.start_run(run_name=run_name):

        # Fit model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        # Test predictions
        predictions = model.predict(test_x)

        # Calculate model signature (input and output)
        signature = infer_signature(train_x, predictions) 

        # Evaluate metrics
        (rmse, mae, r2) = eval_metrics(test_y, predictions)

        # Print out ElasticNet model metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE:", rmse)
        print("  MAE:", mae)
        print("  R2:", r2)

        # Log hyperparams
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log Model
        mlflow.sklearn.log_model(model, "model", signature=signature)
    
        # Log plot as artifact
        image = plot_enet_descent_path(X, y, l1_ratio)
        mlflow.log_artifact("ElasticNet-paths.png")

# COMMAND ----------

# MAGIC %md ### Experiment with different hyperparameters
# MAGIC
# MAGIC * Call `train_diabetes` with different parameters. 
# MAGIC * You can visualize all these runs in the MLflow experiment.

# COMMAND ----------

# MAGIC %md ##### Run 1

# COMMAND ----------

train_diabetes(data, 
    alpha=0.01, 
    l1_ratio=0.01
)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ##### Run 2

# COMMAND ----------

train_diabetes(data, 
    alpha=0.01, 
    l1_ratio=0.75
)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ##### Run 3

# COMMAND ----------

train_diabetes(data, 
    alpha=0.01, 
    l1_ratio=0.5
)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ##### Run 4

# COMMAND ----------

train_diabetes(data, 
    alpha=0.01, 
    l1_ratio=1.0
)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ### View the experiment, run, and notebook  in the MLflow UI
# MAGIC To view the results, click **Experiment** at the upper right of this page. The Experiments sidebar appears. This sidebar displays the parameters and metrics for each run of this notebook. Click the circular arrows icon to refresh the display to include the latest runs. 
# MAGIC
# MAGIC To view the notebook experiment, which contains a list of runs with their parameters and metrics, click the square icon with the arrow to the right of **Experiment Runs**. The Experiment page displays in a new tab. The **Source** column in the table contains a link to the notebook revision associated with each run.
# MAGIC
# MAGIC To view the details of a particular run, click the link in the **Start Time** column for that run. Or, in the Experiments sidebar, click the icon at the far right of the date and time of the run. 
# MAGIC
# MAGIC For more information, see **View notebook experiment** ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).

# COMMAND ----------

display_experiment_uri(experiment)
