"""Causality check of the feature pipeline.

For every record of both benchmarks and every inspection k that is a prediction
point or a training point, the features are recomputed from the record truncated
after acquisition k and compared with row k of the features of the full record.
A causal pipeline gives identical rows: nothing at k may depend on later
acquisitions or on the record's total length. The elapsed-time scale is held
fixed (1.0) because it is a training constant, not a function of the record.

    python code/check_causal_features.py        # exits non-zero on any mismatch
"""
import sys

import numpy as np

import rul_dataset as D
import study_regression as SR


def main():
    worst, n_checked, bad = 0.0, 0, []
    for ds in ("pronostia", "xjtu"):
        full = D.load(ds, t_scale=1.0)
        for b, rec in full.items():
            pts, _ = SR._eval_points(rec, ds, b)
            k0 = max(20, rec["nh"], int(0.05 * rec["n"]))
            train = list(range(k0, rec["n"] - 1, max(1, (rec["n"] - 1 - k0) // 8)))
            for k in sorted(set(pts) | set(train)):
                Xk = D.build_bearing(b, ds, 1.0, n_max=k + 1)[0][k]
                d = float(np.max(np.abs(Xk - rec["X"][k])))
                worst = max(worst, d)
                n_checked += 1
                if d > 1e-9:
                    bad.append((ds, b, k, d))
    print(f"checked {n_checked} (record, inspection) pairs; largest difference {worst:.2e}")
    for x in bad[:10]:
        print("  MISMATCH", x)
    if bad:
        print(f"{len(bad)} mismatches: the pipeline is NOT causal")
        sys.exit(1)
    print("every feature row equals its value computed from the truncated record")


if __name__ == "__main__":
    main()
