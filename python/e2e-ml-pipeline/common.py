import json
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "../../data/train/wine-quality-white.csv"
_name = "e2e-ml-pipeline"
experiment_name = _name
model_name = _name
model_uri = f"models:/{model_name}/production"
docker_image = f"sm-{_name}"
port = 5001

def build_data(data_path):
    col_label = "quality"
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.30, random_state=42)
    X_train = train.drop([col_label], axis=1)
    X_test = test.drop([col_label], axis=1)
    y_train = train[[col_label]]
    y_test = test[[col_label]]
    return X_train, X_test, y_train, y_test

""" Convert CSV file to Pandas JSON format for MLflow scoring server """
def to_json(data_path, num_lines=3):
    with open(data_path, "r") as f:
        lines = f.read().splitlines()
    lines = [ x.split(",") for x in lines ]
    lines = [ x[:-1] for x in lines ] # drop label 'quality'
    columns = [ h.replace('"','')  for h in lines[0] ]
    data = lines[1:1+num_lines] # grab first few data lines
    data = [ [ float(x) for x in line ] for line in data ] # float-ify 
    dct = { "columns": columns, "data": data }
    return json.dumps(dct)
