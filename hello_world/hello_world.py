import time
from argparse import ArgumentParser
import mlflow
from mlflow.entities import Param,Metric,RunTag

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

experiment_name = "hello_world"
print("experiment_name:",experiment_name)
mlflow.set_experiment(experiment_name)

client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
print("experiment_id:",experiment_id)

now = round(time.time())

def run(alpha, run_origin, log_artifact):
    with mlflow.start_run(run_name=run_origin) as run:
        print("runId:",run.info.run_uuid)
        print("artifact_uri:",mlflow.get_artifact_uri())
        print("alpha:",alpha)
        print("log_artifact:",log_artifact)
        print("run_origin:",run_origin)
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", 0.789)
        mlflow.set_tag("run_origin", run_origin)
        mlflow.set_tag("log_artifact", log_artifact)
        if log_artifact:
            with open("info.txt", "w") as f:
                f.write("Hi artifact")
            mlflow.log_artifact("info.txt")

        params = [ Param("p1","0.1"), Param("p2","0.2") ]
        metrics = [ Metric("m1",0.1,now,0), Metric("m2",0.2,now,0) ]
        tags = [ RunTag("t1","hi1"), RunTag("t2","hi2") ]
        client.log_batch(run.info.run_uuid, metrics, params, tags)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--alpha", dest="alpha", help="alpha", default=0.1, type=float )
    parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="")
    parser.add_argument("--log_artifact", dest="log_artifact", help="Log artifact", type=str, default="False")
    args = parser.parse_args()
    run(args.alpha,args.run_origin,args.log_artifact=="True")
