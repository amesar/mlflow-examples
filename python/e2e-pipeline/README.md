# mlflow-examples - e2e-pipeline

End-to-end demonstation of training to real-time prediction with MLflow scoring server.
Basis of subsequent CI/CD pipeline.
  
## Overview
* [train.py](train.py) - Run several training runs with different hyperparameters.
* [register_model.py](register_model.py) - Find the best run and register it as model version `models:/test-e2e-pipeline/production`.
* [deploy_server.py](deploy_server.py) - Launch a scoring server with best model - either local web server or local SageMaker container.
* [test.py](test.py) - Test to run the above train, register and deploy steps in sequence.

## Setup
```
pip install psutil
pip install pytest
pip install pytest-ordering
```

## Run

Default values - see [common.py](common.py):
  * experiment name: `test-e2e-pipeline`
  * model name: `test-e2e-pipeline`
  * model URI:  `models:/test-e2e-pipeline/production`
  * docker image name: `sm-test-e2e-pipeline`

### Train
```
python train.py 
```

```
Arguments:
  experiment_name: test-e2e-pipeline
  data_path: ../../data/wine-quality-white.csv
Experiment ID: 5
Params: (1, 2, 4, 16)
0.820  1 ae9b699cfc5e42f9bca2bd83f95f9524 5
0.785  2 5ae2374d1b524a4e8051c4fa1d7dbcc8 5
0.759  4 4c3da779aa8d48ff97f0c0ed9cafe47c 5
0.867 16 319b94e45b63483cbc8225d0d3a8d6bc 5
Best run: 0.759 4c3da779aa8d48ff97f0c0ed9cafe47c
```

### Register Model
```
python register_model.py 
```
```
Arguments:
  experiment_name: test-e2e-pipeline
  data_path: ../../data/wine-quality-white.csv
Best run: 4c3da779aa8d48ff97f0c0ed9cafe47c 0.7592585886611769
Found model test-e2e-pipeline
Found 0 versions for model test-e2e-pipeline
Reg Model: <class 'mlflow.entities.model_registry.registered_model.RegisteredModel'> {'_name': 'test-e2e-pipeline', '_creation_time': 1584292037095, '_last_updated_timestamp': 1584302072077, '_description': '', '_latest_version': [<ModelVersion: creation_timestamp=1584302072077, current_stage='None', description='', last_updated_timestamp=1584302072077, name='test-e2e-pipeline', run_id='4c3da779aa8d48ff97f0c0ed9cafe47c', source='file:///Users/ander/work/mlflow/server/local_mlrun/mlruns/5/4c3da779aa8d48ff97f0c0ed9cafe47c/artifacts/sklearn-model', status='READY', status_message='', user_id='', version='35'>]}
Version: id=36 status=READY state=None
Waited 0.01 seconds
Version: id=36 status=READY state=None
Version: id=36 status=READY state=Production
predictions: [6.24342105 6.24342105 6.68112798 ... 6.68112798 5.94352941 5.35624284]
```

### Deploy

This script fetches the best model from the model registry and launches an MLflow socring server.
By default it launches a local web server - see [mlflow models serve](https://mlflow.org/docs/latest/cli.html#mlflow-models-serve). 
If you specify the `launch_container` it will create a local SageMaker container - see [mlflow.sagemaker](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow-sagemaker).

#### Run local webserver

Executes the following mlflow commands:
  * `mlflow models serve --port 5001 --model-uri models:/test-e2e-pipeline/production`

```
python -u deploy_server.py 
```
```
Arguments:
  port: 5001
  docker_image: sm-test-e2e-pipeline
  model_uri: models:/test-e2e-pipeline/production
  launch_container: False
  data_path: ../../data/wine-quality-white.csv

Command: mlflow models serve --port 5001 --model-uri models:/test-e2e-pipeline/production
Process ID: 27572
Trying again: 4/10000
. . . 
Trying again: 8/10000
Ran pip subprocess with arguments:
['/Users/andre/miniconda/envs/mlflow-9373994feef365bde89b7072d1498bdbc226ba90/bin/python', '-m', 'pip', 'install', '-U', '-r', '/Users/ander/work/mlflow/server/local_mlrun/mlruns/2/868ce3ccb7c448fe9939c5b21c29ff0a/artifacts/sklearn-model/condaenv.kdzq1ifu.requirements.txt']
Processing /Users/ander/Library/Caches/pip/wheels/6d/72/87/348958818bec20c3a64243396065e34600ada290199f96abfa/mlflow-1.7.0-py3-none-any.whl
Collecting cloudpickle==1.2.2
  Using cached cloudpickle-1.2.2-py2.py3-none-any.whl (25 kB)
. . . 
2020/03/15 22:44:36 INFO mlflow.models.cli: Selected backend for flavor 'python_function'
Done waiting - OK - successful deploy - predictions: [5.335031847133758, 5.050955414012739, 5.726950354609929]
Done waiting - killing process 27572
```

#### Run local SagemMaker container

Executes the following mlflow commands:
  * `mlflow sagemaker build-and-push-container --build --no-push --container sm-test-e2e-pipeline`
  * `mlflow sagemaker run-local -m models:/test-e2e-pipeline/production -p 5001 --image sm-test-e2e-pipeline`

```
python -u deploy_server.py --launch_container
```
```
Starting command: mlflow sagemaker build-and-push-container --build --no-push --container sm-test-e2e-pipeline
. . .
Successfully built a03004c4fa63
Successfully tagged sm-test-e2e-pipeline:latest
Done waiting for command: mlflow sagemaker build-and-push-container --build --no-push --container sm-test-e2e-pipeline
Starting command: mlflow sagemaker run-local -m models:/test-e2e-pipeline/production -p 5001 --image sm-test-e2e-pipeline
2020/03/15 22:48:34 INFO mlflow.models.docker_utils: Building docker image with name sm-test-e2e-pipeline
/var/folders/_9/tbkxzw0116v2cp_zq4f1_1cm0000gp/T/tmp0q3cm6g9/
/var/folders/_9/tbkxzw0116v2cp_zq4f1_1cm0000gp/T/tmp0q3cm6g9//Dockerfile
Sending build context to Docker daemon  3.072kB
Process ID: 27632
Calling scoring server: 0/10000
. . .
Calling scoring server: 23/10000
. . .
Done waiting - OK - Successful deploy - predictions: [5.335031847133758, 5.050955414012739, 5.726950354609929]
Done waiting - killing process 27632
```

## Tests

There are three tests run in sequential order - train, register and deploy.

```
py.test -v test.py
```

```
test.py::test_train PASSED 
test.py::test_register_model PASSED 
test.py::test_deploy_server PASSED 
============================== 3 passed in 54.17s ===============================
```
