# Databricks notebook source
# train_diabetes
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

# COMMAND ----------

import mlflow
mlflow.__version__

# COMMAND ----------

import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

#import from mlflow.models.signature import infer_signature


# COMMAND ----------

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame 
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
train_x = train.drop(["progression"], axis=1)
test_x = test.drop(["progression"], axis=1)
train_y = train[["progression"]]
test_y = test[["progression"]]

# COMMAND ----------

display(train_x)

# COMMAND ----------

display(train_y)

# COMMAND ----------

def train(alpha, l1_ratio):
    print(f"alpha={alpha} l1_ratio={l1_ratio}")
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(train_x, train_y)
    return model

# COMMAND ----------

# alpha and l1_ratio values

params_list = [ 
    (0.01, 0.01),
    (0.01, 0.75),
    (0.01, .5),
    (0.01, 1)
]

# COMMAND ----------

for p in params_list:
    train(p[0], p[1])

# COMMAND ----------

l1_ratio = 0.01
alpha = 0.05
  
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
model.fit(train_x, train_y)
model

# COMMAND ----------

predicted_qualities = model.predict(test_x)
display(predicted_qualities)

# COMMAND ----------


