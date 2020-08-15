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

def main(uri, data_path, num_records, log_mod):
    records = read_data(data_path, num_records)
    headers = { 'Content-Type' : 'application/json' }

    durations = []
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

    print("Results (seconds):")
    print("  mean:   ", round(mean,3))
    print("  max:    ", round(max(durations),3))
    print("  min:    ", round(min(durations),3))
    print("  std:    ", round(stdev,3))
    print("  rsd:    ", round(rsd,2))
    print("  total:  ", round(total,3))
    print("  records:",len(records))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--uri", dest="uri", help="URI", required=True, type=str)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--num_records", dest="num_records", help="num_records", type=int, default=None)
    parser.add_argument("--log_mod", dest="log_mod", help="log_mod", default=100, type=int)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    main(args.uri, args.data_path, args.num_records, args.log_mod)
