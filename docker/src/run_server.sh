
run() {
  STORE_URI=mysql://root:${MYSQL_ROOT_PASSWORD}@db/mlflow
  echo "STORE_URI=$STORE_URI"
  echo "MLFLOW_ARTIFACT_URI=$MLFLOW_ARTIFACT_URI"
  mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $STORE_URI --default-artifact-root $MLFLOW_ARTIFACT_URI 
}

run $* 2>&1 | tee server.log
