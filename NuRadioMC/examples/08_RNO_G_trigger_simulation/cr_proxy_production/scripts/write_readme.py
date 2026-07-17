"""Write a dataset README into the output directory.

Reads each energy bin's manifest.txt (chunks kept) and reports thrown / triggered /
kept counts per bin. No external registry is touched; the README is the only output.
"""
from pathlib import Path
from datetime import datetime


def parse_manifest_header(path):
    """Return the manifest header key=value pairs as ints."""
    with open(path) as f:
        header = f.readline().strip().lstrip("#").strip()
    parts = dict(p.split("=", 1) for p in header.split() if "=" in p)
    return {k: int(v) for k, v in parts.items()}


def main():
    manifests = list(snakemake.input.manifests)
    energies = list(snakemake.params.energies)
    target = int(snakemake.params.target)
    station_id = snakemake.params.station_id
    interaction = snakemake.params.interaction
    out = Path(snakemake.output.readme)

    rows = []
    grand = {"thrown": 0, "trig": 0, "kept": 0, "chunks_total": 0, "chunks_kept": 0}
    for e in energies:
        manifest_path = next((Path(m) for m in manifests if f"lgE{e}/manifest.txt" in m), None)
        if manifest_path is None:
            continue
        h = parse_manifest_header(manifest_path)
        bin_dir = manifest_path.parent
        n_chunks_total = len(sorted(bin_dir.glob("*_ledger.csv")))
        rows.append({
            "lgE": e,
            "thrown": h["all_thrown"],
            "trig": h["all_triggered"],
            "kept": h["kept_triggered"],
            "chunks_total": n_chunks_total,
            "chunks_kept": h["n_kept"],
            "rate": h["all_triggered"] / h["all_thrown"] if h["all_thrown"] else 0.0,
        })
        grand["thrown"] += h["all_thrown"]
        grand["trig"] += h["all_triggered"]
        grand["kept"] += h["kept_triggered"]
        grand["chunks_total"] += n_chunks_total
        grand["chunks_kept"] += h["n_kept"]

    md = []
    md.append(f"# RNO-G FLOWER-trigger simulation dataset (station {station_id}, "
              f"interaction {interaction})\n")
    md.append(f"Simulation for RNO-G station {station_id}, interaction type "
              f"`{interaction}`, produced by the `cr_proxy_production/` Snakemake workflow with "
              f"`simulate.py` (measured FT-noise injection). Contains "
              f"**{grand['kept']:,} kept triggered events** (per-bin triggered targets; "
              f"see the Kept column) across {len(rows)} bins.\n")

    md.append("## Dataset summary\n")
    md.append("| lgE | Thrown | Triggered (all) | Trig/thrown | Kept (target) | Chunks total | Chunks kept |")
    md.append("|-----|--------|-----------------|-------------|----------------|--------------|-------------|")
    for r in rows:
        md.append(f"| {r['lgE']} | {r['thrown']:,} | {r['trig']:,} | {r['rate']*100:.2f}% | "
                  f"{r['kept']:,} | {r['chunks_total']} | {r['chunks_kept']} |")
    md.append(f"| **Total** | **{grand['thrown']:,}** | **{grand['trig']:,}** | | "
              f"**{grand['kept']:,}** | **{grand['chunks_total']}** | **{grand['chunks_kept']}** |\n")

    md.append("## Layout\n")
    md.append("```")
    if rows:
        e = rows[0]["lgE"]
        md.append(f"lgE{e}/")
        md.append(f"  lgE{e}_c0000.nur        # waveforms")
        md.append(f"  lgE{e}_c0000.hdf5       # generator input event list")
        md.append(f"  lgE{e}_c0000_ledger.csv # per-event outcome (status column)")
        md.append("  ...")
        md.append(f"  manifest.txt            # chunk basenames kept for the {target:,}-trigger sample")
    md.append("```\n")
    md.append("Chunks beyond the manifest are kept on disk as overage. To load the "
              "analysis sample, read only the NUR files listed in each bin's "
              "`manifest.txt`.\n")

    md.append("## Provenance\n")
    md.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} via the "
              f"`cr_proxy_production/` Snakemake workflow.\n")
    md.append(f"- Sim script: `simulate.py` with `--interaction_type {interaction}` and "
              f"measured FT-noise injection (`--ft_noise_dir`, `--trigger_vrms`).\n")
    md.append("- Per-event ledger `status` is one of `triggered`, `trigger_failed`, "
              "`efield_cut`; the manifest counts `triggered`.\n")
    md.append("- Chunk IDs may be NON-CONTIGUOUS: scheduler preemption can drop chunks "
              "mid-run. Truncation consumes the on-disk (globbed) ledger set rather "
              "than a contiguous range, so `Chunks total` above is the on-disk count "
              f"per bin, and every bin is truncated to the same {target:,}-trigger "
              "target (see the Kept column).\n")
    md.append("- Near-threshold rate caveat: chunk counts sized by extrapolating the "
              "higher-energy trigger-rate trend are unreliable at the lowest bin, where "
              "the trigger rate falls steeply through threshold (a suppressed-channel "
              "trigger model lowers it further). Size the lowest bin from a measured "
              "production rate.\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"Wrote {out}")


main()
