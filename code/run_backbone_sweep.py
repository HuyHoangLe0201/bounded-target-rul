"""Run the head comparison across several backbones, one after another.

The representation-independence check so far used ONE alternative architecture,
which cannot support a claim about representations in general.  This runs a
sweep.  Every arm uses the SAME epoch budget so that width is the only thing
varying -- the paper's reported configuration (48-24 at 250 epochs) is left
untouched and is not part of the sweep.

Sequential, not parallel: these are CPU-bound and running them together would
only make each slower and the timings meaningless.

    python code/run_backbone_sweep.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "results", "regression")

EPOCHS = "300"
ARMS = [
    ("32,16", "_w32"),        # smaller than the reported backbone
    ("48,24", "_w48"),        # reported width, at the sweep's epoch budget
    ("96,48", "_w96"),        # between
    ("256,128", "_w256"),     # widest, two layers
    ("128,64,32", "_big"),    # three layers
]

for hidden, tag in ARMS:
    out = os.path.join(LOGS, f"run{tag}.log")
    if os.path.exists(os.path.join(LOGS, f"results_both{tag}.json")):
        print(f"skip {hidden} (results_both{tag}.json exists)")
        continue
    env = dict(os.environ, GBS_HIDDEN=hidden, GBS_EPOCHS=EPOCHS, GBS_TAG=tag)
    t0 = time.time()
    print(f"running {hidden} -> {tag} ...", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u",
                            os.path.join(HERE, "study_regression.py"), "both"],
                           env=env, stdout=f, stderr=subprocess.STDOUT,
                           cwd=HERE)
    print(f"  {hidden}: exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)

print("sweep done")
