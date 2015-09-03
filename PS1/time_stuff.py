import matplotlib
import subprocess
import os
import time
from math import ceil

FNULL = open(os.devnull, 'w')

RANGES = ((2, 100), (2, 10**6), (2, 10**9))
NPS = (1, 2, 4, 8)
CMD = "mpirun -np {np} ./parallel {lo} {hi}"

def system_call(command):
    return subprocess.call(command.split(" "), stdout=FNULL)

results = []

for lo, hi in RANGES:
    print(lo, hi)
    n = int(ceil((hi - lo) / 20.0))
    for np in NPS:
        print(np)
        for x in xrange(20):
            new_hi = min(hi, lo + n * (x + 1))
            print(lo, new_hi)
            
            cmd = CMD.format(np=np, lo=lo, hi=new_hi)
            
            start = time.time()
            system_call(cmd)
            end = time.time()
            
            results.append({
                'time': end - start,
                'whole_range': (lo, hi),
                'part_range': (lo, new_hi),
                'np': np
            })

print(results)

f = open('results.txt', 'w')
f.write(str(results))
f.close()
