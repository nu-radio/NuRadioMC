"""Pick the smallest set of chunks whose summed triggered count reaches the target.

Inputs (from the calling rule):
    snakemake.input.ledgers   : list of ledger CSV paths for one energy bin
    snakemake.params.target   : int, target triggered count
    snakemake.output.manifest : path to write the manifest

Manifest format (newline-separated), header then kept NUR basenames:
    # n_kept=K kept_triggered=T all_triggered=A all_thrown=N target=TG
    lgE17.0_c0000.nur
    lgE17.0_c0001.nur
    ...
"""
import os
import re
from pathlib import Path
import pandas as pd


def chunk_id_from_path(p: str) -> int:
    """Extract the zero-padded chunk id from a ledger filename."""
    m = re.search(r"_c(\d{4,6})_ledger\.csv$", os.path.basename(p))
    if not m:
        raise ValueError(f"Cannot parse chunk_id from {p}")
    return int(m.group(1))


def main():
    ledgers = sorted(snakemake.input.ledgers, key=chunk_id_from_path)
    target = int(snakemake.params.target)
    manifest_path = Path(snakemake.output.manifest)

    rows = []
    cumulative = 0
    kept = []
    for lf in ledgers:
        df = pd.read_csv(lf)
        n_trig = int((df["status"] == "triggered").sum())
        n_thrown = int(len(df))
        rows.append({"ledger": lf, "n_trig": n_trig, "n_thrown": n_thrown})
        if cumulative < target:
            kept.append(os.path.basename(lf).replace("_ledger.csv", ".nur"))
            cumulative += n_trig

    grand_trig = sum(r["n_trig"] for r in rows)
    grand_thrown = sum(r["n_thrown"] for r in rows)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        f.write(f"# n_kept={len(kept)} kept_triggered={cumulative} "
                f"all_triggered={grand_trig} all_thrown={grand_thrown} "
                f"target={target}\n")
        for nur in kept:
            f.write(nur + "\n")

    print(f"[truncate {manifest_path}]")
    print(f"  ledgers      : {len(ledgers)}")
    print(f"  total thrown : {grand_thrown:,}")
    print(f"  total trig   : {grand_trig:,}")
    print(f"  target       : {target:,}")
    print(f"  kept chunks  : {len(kept)} (>= target satisfied: {cumulative >= target})")
    print(f"  kept trig    : {cumulative:,}")
    if cumulative < target:
        print(f"  [WARN] kept triggers ({cumulative}) below target ({target}). "
              f"Bump safety_margin or thrown_per_chunk and re-run.")


main()
