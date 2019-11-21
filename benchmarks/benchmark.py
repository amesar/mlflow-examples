import sys
import time
import json
import requests
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", dest="host", help="host", default="localhost")
    parser.add_argument("--port", dest="port", help="port", default=5001)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../data/wine-quality-white.csv")
    parser.add_argument("--num_records", dest="num_records", help="num_records", type=int, default=None)
    parser.add_argument("--log_mod", dest="log_mod", help="log_mod", default=100, type=int)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    with open(args.data_path, 'r') as f:
        lines = f.readlines()
    columns = lines[0].split(',')[:-1] # remove label column 'quality'
    columns = [ c.replace('"','') for c in columns]
    print("Columns:",columns)
    lines = lines[1:]
    if args.num_records is not None:
        lines = lines[:args.num_records]
    num_records = len(lines)
    print("#Records:",num_records)

    records = []
    for line in lines:
        toks = line.strip().split(',')[:-1]
        r = [float(t) for t in toks ]
        dct = { "columns" : columns, "data" : [r] }
        records.append(dct)

    headers = { 'Content-Type' : 'application/json' }
    uri = f"http://{args.host}:{args.port}/invocations"
    durations = []
    print("Calls:")
    for j,r in enumerate(records):
        data = json.dumps(r)
        start = time.time()
        rsp = requests.post(uri, headers=headers, data=data)
        dur = time.time()-start
        if j % args.log_mod == 0:
           print(f"  {j}/{num_records}: {round(dur,3)} - {rsp.text}")
        durations.append(dur)

    total = sum(durations)
    print("Results (seconds):")
    print("  mean:   ", round(total/len(durations),3))
    print("  max:    ", round(max(durations),3))
    print("  min:    ", round(min(durations),3))
    print("  total:  ",round(total,3))
    print("  records:",len(records))
