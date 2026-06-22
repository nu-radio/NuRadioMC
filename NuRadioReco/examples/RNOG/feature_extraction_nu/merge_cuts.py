import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--infile", default = None, required=True)
parser.add_argument("--airplane_out", default = None)
parser.add_argument("--wind_out", default = None)
parser.add_argument("--intrarun_out", default = None)
parser.add_argument("--spatiotemporal_out", default = None)
parser.add_argument("--outfile", default = None, required=True)
args = parser.parse_args()

def run_num(run):
    nums = []
    for char in run:
        if (char.isdigit() == True):
            nums.append(char)
    num = 0

    for i in range(len(nums)):
        num += float(nums[i])*(10**(len(nums)-i-1))
    return int(num)

if (args.spatiotemporal_out is not None):
    infile = pd.read_csv(args.spatiotemporal_out, low_memory=False)
else:
    infile = pd.read_csv(args.infile)

if "spatiotemporal_passed" in infile.columns:
    infile["spatiotemporal_passed"] = (
        infile["spatiotemporal_passed"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .map({"true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )


infile["run_num_clean"] = (
    infile["run_num"].apply(run_num)
)

df_end = infile 
if (args.airplane_out is not None):
    air = pd.read_csv(args.airplane_out).drop(columns=["Unnamed: 0"], errors="ignore")
    df_end = df_end.merge(air, left_on=["run_num_clean", "event_id"], right_on=["run_number", "event_number"],how="left").drop(columns=["run_number", "event_number"], errors="ignore")
if (args.wind_out is not None):
    wind = pd.read_csv(args.wind_out).drop(columns=["Unnamed: 0"], errors="ignore")
    df_end = df_end.merge(wind, left_on=["run_num_clean", "event_id"], right_on=["run_number", "event_number"],how="left").drop(columns=["run_number", "event_number"], errors="ignore")
if (args.intrarun_out is not None):
    intrarun = pd.read_csv(args.intrarun_out).drop(columns=["Unnamed: 0"], errors="ignore")
    df_end = df_end.merge(intrarun, left_on=["run_num_clean", "event_id"], right_on=["run_number", "event_number"],how="left").drop(columns=["run_number", "event_number"], errors="ignore")

df_end = df_end.drop(columns=["run_num_clean"])

df_end.to_csv(args.outfile, index=False)

