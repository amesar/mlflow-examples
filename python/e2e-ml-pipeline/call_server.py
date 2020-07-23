import requests
import common

def run(api_uri, data_path):
    data = common.to_json(data_path)
    return call(api_uri, data)

def call(api_uri, data):
    headers = { "Content-Type": "application/json" }
    rsp = requests.post(api_uri, headers=headers, data=data)
    print(f"call: status_code={rsp.status_code}")
    if rsp.status_code < 200 or rsp.status_code > 299:
        return None
    return rsp.text

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--api_uri", dest="api_uri", help="API URI", default="http://localhost:{common.port}")
    parser.add_argument("--data_path", dest="data_path", help="Data path", default=common.data_path)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    rsp = run(args.api_uri, args.data_path)
    print(f"Predictions: {rsp}")
