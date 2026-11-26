---
name: nuradio-tests
description: "Use when adding or changing a NuRadio test, or when a change alters physics output and reference-comparison tests fail."
---

# NuRadio tests

CI runs exactly the steps listed in `.github/workflows/run_tests.yaml`, each from the repository root. `run_all_tests.sh` covers only the shell-driven subset.

## Adding a test

1. Put it next to its topic in `NuRadioMC/test/` or `NuRadioReco/test/`, as a script that exits non-zero on failure. Seed every random generator.
2. Add a step to the matching job in `run_tests.yaml` (`- name: "..."` with `run: |` and the command, path relative to the repo root). Without the step, CI never runs it.
3. Files the test writes are removed at its end (see `NuRadioMC/test/SingleEvents/test_build.sh`) and stay out of git.
4. Done when the test passes from the repo root and goes red when the tested code is broken (break it temporarily to check).

## Reference tests

Pattern: a run script simulates with fixed seeds, a validate script compares the output with a committed reference file within tolerances (e.g. `NuRadioMC/test/SingleEvents/T04validate_allmost_equal.py`, `--rtol`).

When one fails after an intended physics change:

1. Quantify the change: before/after numbers for the affected quantities (Veff, trigger rate, amplitudes). They go into the PR description.
2. Regenerate the reference the way that test provides: look in its scripts for a create option (e.g. `NuRadioReco/test/trigger_tests/compare_to_reference.py --create_reference`, `NuRadioMC/test/SingleEvents/test_build.sh --create_reference`); otherwise rerun the run script and replace the reference file with its output.
3. Commit the new reference separately, with the justification in the message. Keep tolerances unchanged unless the PR argues for the new value.

A failing reference test after a change meant to keep physics output identical (refactor, speed-up) is a bug in the change.
