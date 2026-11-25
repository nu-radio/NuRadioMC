---
name: nuradio-pr-review
description: "Use when reviewing a NuRadio pull request."
---

# Reviewing a NuRadio PR

1. Read the discussion first; reproducers and context often live there:
   `gh pr view <n> -R nu-radio/NuRadioMC --comments` and the inline threads with `gh api repos/nu-radio/NuRadioMC/pulls/<n>/comments`.
2. Diff with `gh pr diff <n> -R nu-radio/NuRadioMC`. It diffs against the PR's own base, so a PR stacked on another open PR shows only its own changes.
3. Go through the checklist below. Done when every item is either confirmed or recorded as a finding with `file:line`.
4. To run code, check out the PR head in a separate worktree or clone and run the tests the diff touches (see the `nuradio-tests` skill).

## Checklist

- Targets `develop`; `changelog.txt` has a line under the current `-dev` version.
- New public functions and classes have numpydoc docstrings; non-importable names use double backticks.
- Public API is deprecated, not removed or renamed (`deprecated` in `NuRadioReco/utilities/logging.py`).
- New `framework/parameters.py` enum members are appended at the bottom with new values and a `#:` comment.
- Units via `NuRadioReco.utilities.units`, FFTs via `NuRadioReco.utilities.fft`.
- Random numbers come from a dedicated, seedable generator made by `get_rng` (`NuRadioReco/utilities/random_generator.py`; Conventions in `AGENTS.md`). Each use of numpy's global random state, `RandomState`, stdlib `random`, or an unseeded `default_rng()` without a `seed`/`rng` parameter in new or changed code is a finding.
- New modules: `run(evt, station, det, ...)` with `@register_run()`, plus `begin` and `end`.
- New tests are added as steps in `.github/workflows/run_tests.yaml`.
- Physics-output changes come with before/after numbers and justified reference updates.
- Python 3.8 compatible (PR CI runs only 3.10): no `match`, no runtime `X | Y` type unions, no newer stdlib APIs.
- New dependencies are flagged for discussion with the maintainers.
