import sys
import time
import json
import threading
import requests
from argparse import ArgumentParser
import common 

class MyThread(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self, args=args)
        self.records = args[0]
        self.idx = args[1]
        self.mean = -1
        self.max = -1
        self.min = -1

    def run(self):
        log_filename = f"run_{self.idx}.log"
        with open(log_filename, 'w') as f:
            return self._run(f)

    def _run(self,f):
        uri = f"http://{args.host}:{args.port}/invocations"
        headers = { 'Content-Type' : 'application/json' }
        durations = []
        f.write(f"Calls for thread {self.idx}:\n")
        num_records = len(self.records)
        for j,r in enumerate(self.records):
            data = json.dumps(r)
            start = time.time()
            rsp = requests.post(uri, headers=headers, data=data)
            dur = time.time()-start
            if j % args.log_mod == 0:
               f.write(f"  {j}/{num_records}: {round(dur,3)}\n")
               sys.stdout.write(f" thr_{self.idx}  {j}/{num_records}: {round(dur,3)}\n")
            durations.append(dur)

        total = sum(durations)
        self.mean = total/len(durations)
        self.max = max(durations)
        self.min = min(durations)

        f.write(f"Results (seconds):")
        f.write(f"  mean:    {round(self.mean,3)}\n")
        f.write(f"  max:     {round(self.max,3)}\n")
        f.write(f"  min:     {round(self.min,3)}\n")
        f.write(f"  total:   {round(total,3)}\n")
        f.write(f"  records: {len(records)}\n")

    def get_stats(self):
        return self.mean, self.max, self.min

def fmt(x):
    y = round(x,3)
    return str(y).ljust(5, '0')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", dest="host", help="host", default="localhost")
    parser.add_argument("--port", dest="port", help="port", default=5001)
    parser.add_argument("--data_path", dest="data_path", help="data_path", default="../../data/train/wine-quality-white.csv")
    parser.add_argument("--num_records", dest="num_records", help="num_records", type=int, default=None)
    parser.add_argument("--log_mod", dest="log_mod", help="log_mod", default=100, type=int)
    parser.add_argument("--num_threads", dest="num_threads", help="num_threads", type=int, default=1)
    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    records = common.read_data(args.data_path, args.num_records)

    threads = []
    for j in range(args.num_threads):
        t = MyThread(args=(records,j,))
        threads.append(t)
        t.start()
    print(f"Spawned {args.num_threads} threads")
    for t in threads:
        t.join()

    print("Summary")
    print(f"  Mean  Max   Min")
    for t in threads:
       mean,max,min = t.get_stats()
       print(f"  {fmt(mean)} {fmt(max)} {fmt(min)}")
