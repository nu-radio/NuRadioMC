import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sim_chunks", nargs="+", default = None)
parser.add_argument("--burn_chunks", nargs="+", default = None)
parser.add_argument("--out_sim", default = None)
parser.add_argument("--out_burn", default = None)
args = parser.parse_args()

if (args.sim_chunks is not None):
    sim_df = pd.concat([pd.read_csv(f) for f in args.sim_chunks])
    sim_df.to_csv(args.out_sim, index=False)
if (args.burn_chunks is not None):
    burn_df = pd.concat([pd.read_csv(f) for f in args.burn_chunks])
    burn_df.to_csv(args.out_burn, index=False)
