
# MLflow Docker - MLflow Tracking Server and MySQL 

## Overview

Launches a full-fledged MLflow server enviroment consisting of two containers:
* mlflow_server - MLflow tracking server
* mlflow_mysql - MySQL database server

Two types of mlflow_server containers can be built depending on where the artifact repository lives:
  * local - mounted shared volume between (laptop) host and container 
  * S3 - artifacts are stored on S3

See [MLflow Tracking Servers](https://mlflow.org/docs/latest/tracking.html#mlflow-tracking-servers) and
[Referencing Artifacts](https://mlflow.org/docs/latest/concepts.html#referencing-artifacts).


## Run

The required  environment variables are specified in the standard docker compose `.env` file.
Copy one of the two `.env` template files to `.env`, make appropriate changes and then run docker-compose.

To launch a local MLflow server:
```
cp .env-local-template .env
vi .env # make changes 
docker-compose -f docker-compose.yaml -f docker-compose-local.yaml  up -d 
```

To launch an S3 MLflow server:
```
cp .env-s3-template .env
vi .env # make changes 
docker-compose -f docker-compose.yaml -f docker-compose-s3.yaml up -d 
```
You will then see two containers:
```
CONTAINER ID  IMAGE                  COMMAND                  PORTS                     NAMES
7a4be1019858  mlflow_server:latest   "/bin/sh -c 'cd /hom…"   0.0.0.0:5005->5000/tcp    mlflow_server
3b4eb5a2026e  mlflow_mysql:5.7.31    "docker-entrypoint.s…"   0.0.0.0:33306->3306/tcp   mlflow_mysql
```
If you don't see the `mlflow_server` container, just run the docker-compose command again. 
It failed to start because `mlflow_mysql` wasn't up yet. It's a TODO to add a wait-until-alive feature.

## Environment variables

| Env Var  | Description  | Default  |
|:--|:--|:--|
| **MySQL**  |   |   |
| MYSQL_ROOT_PASSWORD | MySQL root password  | efcodd   |
| HOST_MYSQL_PORT  | Port exposed on host  | 5306  |
|  HOST_MYSQL_DATA_DIR  | Host mounted volume path |   |
| **MLflow**  |   |   |
| MLFLOW_ARTIFACT_URI  | Base URI for artifacts - either S3 or local path|   |
| HOST_MLFLOW_PORT  | Port exposed on host for tracking server  | 5005  |
| **MLflow S3**  |   |   |
| AWS_ACCESS_KEY_ID  |   |   |
| AWS_SECRET_ACCESS_KEY  |   |   |


**Sample local .env**
```
# MySQL 
MYSQL_ROOT_PASSWORD=efcodd
HOST_MYSQL_PORT=5306
HOST_MYSQL_DATA_DIR=/opt/mlflow_docker/mysql

# MLflow tracking server
MLFLOW_ARTIFACT_URI=/opt/mlflow_docker/mlflow_server
HOST_MLFLOW_PORT=5005
```

**Sample S3 .env**
```
# MySQL 
MYSQL_ROOT_PASSWORD=efcodd
HOST_MYSQL_PORT=5306
HOST_MYSQL_DATA_DIR=/opt/mlflow_docker/mysql

# MLflow tracking server
MLFLOW_ARTIFACT_URI=s3://my-bucket/mlflow
HOST_MLFLOW_PORT=5005

# AWS 
AWS_ACCESS_KEY_ID=my_access_key_id
AWS_SECRET_ACCESS_KEY=my_secret_access_key
```

## Check server health

**Database**
```
docker exec -it mlflow_mysql mysql \
  -u root --password=efcodd --port=5306 \
  -e "use mlflow ; select * from experiments"
```
```
+---------------+---------+-------------------------------------+-----------------+
| experiment_id | name    | artifact_location                   | lifecycle_stage |
+---------------+---------+-------------------------------------+-----------------+
|             0 | Default | /opt/mlflow_docker/mlflow_server/0  | active          |
+---------------+---------+-------------------+-----------------+-----------------+
```

**Tracking server**
```
curl http://localhost:5005/api/2.0/mlflow/experiments/list
```
```
{
  "experiments": [
    {
      "experiment_id": "0",
      "name": "Default",
      "artifact_location": "/opt/mlflow_docker/mlflow_server/0",
      "lifecycle_stage": "active"
    }
  ]
```

## Login to containers

You can check things out inside the containers.
```
docker exec -i -t mlflow_server /bin/bash
```
```
docker exec -i -t mlflow_mysql /bin/bash
```

## Files

|   | Dockerfile   | Compose | Note |
|---|---|---|--|
| General  |  | [docker-compose.yaml](docker-compose.yaml)  | Basic config for MySQL and MLflow server |
|MySQL  | [Dockerfile-mysql](Dockerfile-mysql)  | | MySQL-specific configs |
| MLflow Local  |[Dockerfile-mlflow-local](Dockerfile-mlflow-local)  |[docker-compose-local.yaml](docker-compose-local.yaml)  | Local-specific configs|
| MLflow S3 | [Dockerfile-mlflow-s3](Dockerfile-mlflow-s3)  |[docker-compose-s3.yaml](docker-compose-s3.yaml)  | S3-specific configs|
