import time
import json
import requests
import statistics
from argparse import ArgumentParser
from common import read_data

import platform
print("Versions:")
print("  platform:", platform.platform())
print("  python_version:", platform.python_version())

def main(uri, data_path, num_records, log_mod, output_file_base, num_iters):
    records = read_data(data_path, num_records)
    headers = { 'Content-Type' : 'application/json' }

    durations = []
    for iter in range(0,num_iters):
        num_records = len(records)
        print("Calls:")
        for j,r in enumerate(records):
            data = json.dumps(r)
            start = time.time()
            requests.post(uri, headers=headers, data=data)
            dur = time.time()-start
            if j % log_mod == 0:
                print(f"  {j}/{num_records}: {round(dur,3)}")
            durations.append(dur)

    total = sum(durations)
    mean = statistics.mean(durations)
    stdev = statistics.stdev(durations)
    rsd = stdev / mean * 100 # relative stdev

    calls = num_iters * len(records)
    print("Results (seconds):")
    print("  mean:   ", round(mean,3))
    print("  max:    ", round(max(durations),3))
    print("  min:    ", round(min(durations),3))
    print("  std:    ", round(stdev,3))
    print("  rsd:    ", round(rsd,2))
    print("  total:  ", round(total,3))
    print("  calls:     ",calls)
    print("  records:   ",len(records))
    print("  iterations:", num_iters)

    if output_file_base:
        now = time.time()
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
        dct = {
          "timestamp": ts,
          "uri": uri,
          "mean": mean,
          "max": max(durations),
          "min": min(durations),
          "std": stdev,
          "rsd": rsd,
          "total": total,
          "calls": calls,
          "records": len(records),
          "iterations": num_iters
        }
        ts = time.strftime("%Y-%m-%d_%H%M%S", time.gmtime(now))
        path = f"{output_file_base}_{ts}.csv"
        print("Output file:",path)
        with open(path, "w") as f:
            f.write(json.dumps(dct,indent=2)+"\n")
   

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--uri", dest="uri", help="URI", required=True, type=str)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--num_records", dest="num_records", help="num_records", type=int, default=None)
    parser.add_argument("--log_mod", dest="log_mod", help="log_mod", default=100, type=int)
    parser.add_argument("--output_file_base", dest="output_file_base", help="Output file base", default=None)
    parser.add_argument("--num_iters", dest="num_iters", help="Number of iterations over data", default=1, type=int)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    main(args.uri, args.data_path, args.num_records, args.log_mod, args.output_file_base, args.num_iters)
