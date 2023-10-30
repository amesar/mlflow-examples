"""
Lightweight CICD pipeline for Notebook Training
"""

import os
import shutil
import time
import json
import requests
import base64
from databricks_cli.sdk.service import WorkspaceService, JobsService, DbfsService
import mlflow
from base_cicd_driver import BaseCicdDriver, fmt

def strip_underscores(obj):
    return { k[1:]:v for (k,v) in obj.__dict__.items() }

class CicdDriver(BaseCicdDriver):
    def __init__(self, profile, src_dir, files, dst_dir, scratch_dir, cluster_spec_file, report_file, args):
        super().__init__(profile, cluster_spec_file, args)
        self.src_dir = src_dir
        self.scratch_dir = scratch_dir
        self.dst_dir = dst_dir
        self.files = files
        self.report_file = report_file
        self.dbfs_service = DbfsService(self.api_client)
        self.jobs_service = JobsService(self.api_client)
        self.workspace_service = WorkspaceService(self.api_client)


    def mk_scratch_path(self, path):
        return os.path.join(self.scratch_dir.replace("dbfs:","/dbfs"), path)

    def mk_dst_path(self, path):
        return os.path.join(self.dst_dir, path)


    def download_from_git(self, uri):
        """ Download notebooks from git """
        filename = os.path.basename(uri)
        print(f"Downloading {uri} to {self.scratch_dir}")
        rsp = requests.get(uri)
        if rsp.status_code != 200:
            raise Exception(f"ERROR: StatusCode: {rsp.status_code} URI: {uri}")
        with open(self.mk_scratch_path(filename),"w") as f:
            f.write(rsp.text)
    

    def download_from_workspace(self, wks_path):
        """ Download notebooks from Databricks workspace """
        wks_path = wks_path.replace(".py","")
        filename = os.path.basename(wks_path)
        print(f"Downloading {wks_path} to {self.scratch_dir}")
        rsp = self.workspace_service.export_workspace(wks_path)
        content = rsp["content"]
        content = base64.b64decode(content).decode()
        with open(self.mk_scratch_path(f"{filename}.py"),"w") as f:
            f.write(content)
    

    def download_notebooks(self):
        """ Download notebooks """
        print("**** Downloading notebooks")
        if os.path.exists(self.scratch_dir): 
            shutil.rmtree(self.scratch_dir)
        os.makedirs(self.scratch_dir, exist_ok=True)
        if self.src_dir.startswith("https://raw.github"):
            for file in files:
                self.download_from_git(f"{self.src_dir}/{file}")
        else:
            for file in files:
                self.download_from_workspace(f"{self.src_dir}/{file}")
        self.report["downloaded_files"] = [ f"{self.src_dir}/{file}" for file in files ]
    

    def import_notebooks(self):
        """ Import notebooks into Databricks workspace """
        print("**** Importing notebooks into Databricks")
        #self.workspace_service.delete(self.dst_dir, True)
        self.workspace_service.mkdirs(self.dst_dir)
        files = os.listdir(self.scratch_dir)
        for file in files:
            ipath = self.mk_scratch_path(file)
            opath = self.mk_dst_path(file).replace(".py","")
            print(f"Importing into {opath}")
            with open(ipath, "r") as f:
                content = f.read()
            content = base64.b64encode(content.encode()).decode()
            self.workspace_service.import_workspace(opath, language="PYTHON", content=content, overwrite=True) # OK for dbx
        self.report["uploaded_files"] = [ self.mk_dst_path(file).replace(".py","") for file in files ]


    def check_run(self, job_run):
        """ Check results of the job run and the MLflow run"""

        print("**** Checking job run")
        # Check Databricks job results
        state = job_run["state"]
        print("state:",job_run["state"])
        life_cycle_state = state["life_cycle_state"] 
        if life_cycle_state == "INTERNAL_ERROR":
            print(f"ERROR: job failed: {state}")
            return
        result_state = state["result_state"]
        self.report["info"]["result_state"] = result_state
        if result_state != "SUCCESS":
            print(f"ERROR: job failed - result_state: {result_state}")
            return

        # Checking MLflow run
        print("**** Checking MLflow run")
        client = mlflow.tracking.MlflowClient()
        print("Tracking URI:", mlflow.tracking.get_tracking_uri())

        # Get MLflow run ID from job output file
        scratch_dir = self.cluster_spec["notebook_task"]["base_parameters"]["Scratch Dir"]
        output_file = os.path.join(scratch_dir,"mlflow_cicd.log")
        print("Output file:",output_file)
        rsp = self.dbfs_service.read(output_file)
        run_id = rsp["data"]
        run_id = base64.b64decode(run_id).decode()

        # Get MLflow run 
        run = client.get_run(run_id)
        experiment = client.get_experiment(run.info.experiment_id)
        run_uri = f"{self.host}/#mlflow/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
        print(f"Run info:")
        print(f"  Experiment ID: {run.info.experiment_id}")
        print(f"  Experiment Name: {experiment.name}")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Run params: {run.data.params}")
        print(f"  Run metrics: {run.data.metrics}")
        print(f"  Run URI: {run_uri}")
        print(f'  Browser: open -a "Google Chrome" "{run_uri}"')

        self.report["mlflow_run"] =  \
            { "run_uri": run_uri,
              "run": { "info": strip_underscores(run.info), 
                       "params": run.data.params, 
                       "metrics": run.data.metrics, 
                       "tags": run.data.tags } }

    
    def run(self):
        self.download_notebooks()
        self.import_notebooks()
        run = self.run_job()
        self.check_run(run)
        self.report["info"]["end_time"] = fmt(time.time()) 
        with open(self.report_file, "w") as f:
            f.write(json.dumps(self.report,indent=2)+"\n")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--profile", dest="profile", help="Profile in ~/.databrickscfg profile", default=None)
    parser.add_argument("--src_dir", dest="src_dir", help="https://raw.github URI or Databricks workspace folder", required=True)
    parser.add_argument("--src_files", dest="src_files", help="Source notebooks - comma delimited", required=True)
    parser.add_argument("--dst_dir", dest="dst_dir", help="Destination Databricks scratch workspace folder", required=True)
    parser.add_argument("--scratch_dir", dest="scratch_dir", help="Temporary scratch folder for downloaded notebooks", default="out")
    parser.add_argument("--cluster_spec_file", dest="cluster_spec_file", help="JSON cluster spec file", required=True)
    parser.add_argument("--report_file", dest="report_file", help="Report file", default="report.json")
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    files = args.src_files.split(",")
    driver = CicdDriver(args.profile, args.src_dir, files, args.dst_dir, args.scratch_dir, args.cluster_spec_file, args.report_file, args)
    driver.run()
