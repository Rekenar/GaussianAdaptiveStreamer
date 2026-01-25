#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_ndjson(fp):
    data = []
    for line in fp:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.strip().startswith("..."):
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            line = line.rstrip(",")
            try:
                obj = json.loads(line)
            except Exception:
                continue
        data.append(obj)
    return data

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, default="-",
                    help="Path to NDJSON file. Use '-' to read from stdin (default).")
    ap.add_argument("--out-prefix", "-o", type=str, default="pred_plot",
                    help="Prefix for saved PNG files.")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open interactive windows; only save PNGs.")
    args = ap.parse_args()

    # Read input
    if args.input == "-" or args.input is None:
        data = load_ndjson(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = load_ndjson(f)

    if not data:
        print("No data parsed. Ensure your input is NDJSON with one object per line.", file=sys.stderr)
        sys.exit(1)

    # Extract and normalize series
    t0 = min(d.get("t_server") for d in data if "t_server" in d)
    times_s = [ (d.get("t_server", t0) - t0) / 1000.0 for d in data ]



    pred_bps = [ safe_float(d.get("pred_bps")) for d in data ]
    tc_status = [ safe_float(d.get("tc_status")) for d in data ]
    profile   = [ safe_int(d.get("profile")) for d in data ]

    # Convert to kbps for readability
    pred_kbps = [ v if v == v else float("nan") for v in pred_bps ]  # v==v filters NaN
    tc_kbps   = [ v / 8 if v == v else float("nan") for v in tc_status ]

    # --- Figure 1: Throughput (kbps) ---
    plt.figure(figsize=(10, 5))
    plt.plot(times_s, pred_kbps, label="pred_bps (kBps)")
    plt.plot(times_s, tc_kbps, label="tc_status (kBps)")
    plt.title("Predicted vs TC Status Throughput")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Throughput (kBps)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    out1 = f"{args.out_prefix}_throughput.png".replace("/", "_")
    plt.tight_layout()
    plt.savefig(out1, dpi=150)

    # --- Figure 2: Profile (step plot) ---
    plt.figure(figsize=(10, 3.5))
    # Use drawstyle='steps-post' for a staircase look without subplots
    plt.plot(times_s, profile, drawstyle="steps-post")
    plt.title("Selected Profile over Time")
    plt.xlabel("Time since start (s)")
    plt.ylabel("Profile")
    plt.yticks(sorted(set(profile)))
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    out2 = f"{args.out_prefix}_profile.png".replace("/", "_")
    plt.tight_layout()
    plt.savefig(out2, dpi=150)

    if args.no_show:
        print(f"Saved: {out1}")
        print(f"Saved: {out2}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
