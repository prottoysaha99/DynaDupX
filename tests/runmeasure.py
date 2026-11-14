#!/usr/bin/env python3
"""runmeasure.py <label> <cmd...> : runs a command, prints wall time, peak RSS of the child, and key output lines."""
import re, resource, subprocess, sys, time
label, cmd = sys.argv[1], sys.argv[2:]
t = time.time()
p = subprocess.run(cmd, capture_output=True, text=True)
el = time.time() - t
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
out = p.stderr + p.stdout
cost = re.search(r"total cost: (-?[0-9.E]+)", out); ev = re.search(r"EVENTS .*", out)
oom = "OutOfMemoryError" in out
states = re.search(r"DP states: (\d+), candidate bipartitions: (\d+)", out)
print("%-28s rc=%d time=%.1fs peakRSS=%.0fMB cost=%s %s %s%s" % (label, p.returncode, el, rss, cost.group(1) if cost else "NA",
      ("states=%s cands=%s" % states.groups()) if states else "", ev.group(0) if ev else "", " OOM" if oom else ""), flush=True)
