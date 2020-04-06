"""
Lightweight CICD pipeline for Notebook Training
"""

from abc import abstractmethod, ABCMeta
#import os
import time
import json
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.sdk.service import WorkspaceService, JobsService, DbfsService
#import mlflow

def fmt(ts):
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(ts))

def get_credentials(profile):
    from databricks_cli.configure import provider
    cfg = provider.get_config() if profile is None else provider.get_config_for_profile(profile)
    return (cfg.host, cfg.token)

class BaseCicdDriver(metaclass=ABCMeta):
    def __init__(self, profile, cluster_spec_file, args):
        (self.host,token) = get_credentials(profile)
        print("Host:",self.host)
        self.api_client = ApiClient(None, None, self.host, token)
        self.report = { "info": { "start_time": fmt(time.time())} }
        self.report["arguments"] = vars(args)
        self.cluster_spec = self.read_cluster_spec(cluster_spec_file)
        self.cluster_spec["run_name"] = self.cluster_spec.pop("name")

    def read_cluster_spec(self, cluster_spec_file):
        with open(cluster_spec_file, "r") as f:
            cluster_spec = json.loads(f.read())
            self.report["cluster_spec_file"] = cluster_spec_file
            self.report["cluster_spec"] = cluster_spec
            return cluster_spec

    def run_job(self):
        """ Run the Databricks job """
        print("**** Running MLflow training job")
        rsp = self.jobs_service.submit_run(**self.cluster_spec)
        run_id = rsp["run_id"]
        print("Run",run_id)
        idx = 0
        start = time.time()
        while True:
            rsp = self.jobs_service.get_run(run_id)
            state = rsp["state"]["life_cycle_state"]
            print(f"{idx}: Run {run_id} waiting {state}")
            if not state in ["PENDING", "RUNNING"]: 
                break
            time.sleep(2)
            idx += 1
        end = time.time()
        print(f"Waited {round(end-start,1)} seconds")
        self.report["databricks_job"] = rsp
        return rsp
