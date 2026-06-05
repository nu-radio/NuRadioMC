import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--sim", required=True)
parser.add_argument("--burn", required=True)
parser.add_argument("--out", required=True)

args = parser.parse_args()

sim_joined = pd.read_csv(args.sim)
burn_joined = pd.read_csv(args.burn)
sim_joined["source"] = "sim"
sim_joined["log10_energy"] = 16.0   
burn_joined["source"] = "burn"

full = pd.concat(
    [burn_joined, sim_joined],
    ignore_index=True,
    sort=False)

full.to_hdf(args.out, key="events", mode="w", complevel=5, complib="zlib", format="table")

