#!/usr/bin/env python3
"""
Checks that every ``ant_type`` in the detector json files of this repository is a known antenna model,
i.e., an analytic model or a model listed in ``NuRadioReco/detector/antenna_models_hash.json``.

Since v3.2, ``ant_type`` is used as is (no suffix like '_InfFirn' is appended), hence a name missing
from the hash file can not be loaded.
"""
import glob
import json
import os

import NuRadioReco

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(NuRadioReco.__file__)))

# Antenna models which are used in detector jsons but do not exist (on the data server). Remove an entry
# once it is fixed; the test fails if an entry is not needed anymore.
KNOWN_MISSING = {"ARA_bicone_InfFirn", "KU_dipole_InfFirn"}


def get_known_antenna_models():
    with open(os.path.join(repo_dir, "NuRadioReco", "detector", "antenna_models_hash.json")) as f:
        return {os.path.splitext(name)[0] for name in json.load(f)}


def get_ant_types():
    """ Returns {ant_type: set of json files using it} for all detector jsons in the repository """
    ant_types = {}
    for package in ["NuRadioMC", "NuRadioReco"]:
        for filename in glob.glob(os.path.join(repo_dir, package, "**", "*.json"), recursive=True):
            with open(filename) as f:
                content = f.read()

            if '"ant_type"' not in content:
                continue

            channels = json.loads(content).get("channels", {})
            for channel in channels.values():
                if "ant_type" in channel:
                    ant_types.setdefault(channel["ant_type"], set()).add(os.path.relpath(filename, repo_dir))

    return ant_types


def test_ant_types_exist():
    known_models = get_known_antenna_models()
    ant_types = get_ant_types()
    assert len(ant_types), "No detector json with `ant_type` found"

    missing = {
        name: files for name, files in ant_types.items()
        if not name.startswith("analytic") and name not in known_models}

    unexpected = {name: files for name, files in missing.items() if name not in KNOWN_MISSING}
    assert not unexpected, "Unknown antenna models (not in antenna_models_hash.json):\n" + "\n".join(
        f"  {name}: {', '.join(sorted(files))}" for name, files in sorted(unexpected.items()))

    fixed = KNOWN_MISSING - set(missing)
    assert not fixed, f"Remove {sorted(fixed)} from KNOWN_MISSING (exist now or are not used anymore)"


if __name__ == "__main__":
    test_ant_types_exist()
    print("Detector json tests passed")
