import sys, time
import psutil
from subprocess import Popen
import mlflow
import common
import call_server

client = mlflow.tracking.MlflowClient()
print(f"MLflow Version: {mlflow.__version__}")
sleep_time = 2
iterations = 10000

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def wait_until_ready(uri, data_path):
    import requests
    data = common.to_json(data_path)
    start = time.time()
    for j in range(iterations): 
        rsp = None
        try:
            rsp = call_server.call(uri, data)
        except requests.exceptions.ConnectionError as e:
            print(f"Calling scoring server: {j}/{iterations}")
        if rsp is not None: 
            print(f"Done waiting for {time.time()-start:5.2f} seconds")
            return rsp
        time.sleep(sleep_time)
    raise Exception(f"ERROR: Timed out after {iterations} iterations waiting for server to launch")

def run_local_webserver(model_uri, port):
    cmd = f"mlflow models serve --port {port} --model-uri {model_uri}"
    print("Command:",cmd)
    cmd = cmd.split(" ")
    return Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True)

def run_local_sagemaker(model_uri, port, docker_image):
    cmd = f"mlflow sagemaker build-and-push-container --build --no-push --container {docker_image}"
    print("Starting command:",cmd)
    proc = Popen(cmd.split(" "), stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True)
    proc.wait()
    print("Done waiting for command:",cmd)

    cmd = f"mlflow sagemaker run-local -m {model_uri} -p {port} --image {docker_image}"
    print("Starting command:",cmd)
    proc = Popen(cmd.split(" "), stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True)
    return proc

def run(model_uri, port, data_path, docker_image, launch_container):
    print(f"==== {__file__} ====")
    if launch_container:
        proc = run_local_sagemaker(model_uri, port, docker_image)
    else:
        proc = run_local_webserver(model_uri, port)
    print(f"Process ID: {proc.pid}")

    uri = f"http://localhost:{port}/invocations"
    try:
        rsp = wait_until_ready(uri, data_path)
        print(f"Done waiting - OK - successful deploy - predictions: {rsp}")
    except Exception as e:
        print("ERROR:",e)
    print(f"Done waiting - killing process {proc.pid}")
    kill(proc.pid)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--port", dest="port", help="port", default=common.port, type=int)
    parser.add_argument("--docker_image", dest="docker_image", help="Docker image name", default=common.docker_image)
    parser.add_argument("--model_uri", dest="model_uri", help="Model URI", default = common.model_uri)
    parser.add_argument("--launch_container", dest="launch_container", help="Launch local SageMaker container instead of local server", default=False, action='store_true')
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=common.data_path)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    run(args.model_uri, args.port, args.data_path, args.docker_image, args.launch_container)
