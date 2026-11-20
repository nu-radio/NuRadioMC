---
name: nuradio-pr
description: "Use when writing or editing a NuRadio pull request title or description."
---

# Writing a NuRadio PR

Base branch: `develop`. Before opening, check that `changelog.txt` has a line under the current `-dev` version ("new features" or "bugfixes").

**Title:** imperative. For a bug fix, name the bug, not the solution (`Fix wrong amplitude units after trace resampling`); otherwise name the effect a user sees.

**Labels:** at least one. Pick from `gh label list`: a component (`NuRadioMC`, `NuRadioReco`, `RNO-G`, `Detector`, ...) plus a type (`Bug`, `New Feature`, `cleanup`, `Documentation/CI/Examples`).

**Body:** this template replaces the text of `pull_request_template.md`. Use the terms of `GLOSSARY.md`. Every claim is something verified; anything else is left out.

```markdown
## Summary

<1-3 sentences: what was wrong or missing, what changes. Link the issue if there is one.>
<optional: diagram, diff sketch, or call tree>

## Evidence

- <test or check run> → <result>
- <size of the effect, if outputs change>
- Not tested: <paths the checks did not exercise>

## Backwards compatibility & downstream impact

- **API:** <unchanged | extended | breaking> — <what calling code must change>
- **Data:** <unchanged | changed> — <existing input files/configs still read? files written still readable by older versions?>
- **Results:** <unchanged | changed> — <which outputs, for which inputs/settings, by how much>
- **Migration:** <what users do to keep old behaviour or adapt; omit if nothing>

<optional: merge order, related PRs/branches, follow-ups>
```

## Summary

State the problem and the fix in the reader's terms: the quantity or behaviour a user sees. Add a visual only when it is clearer than the sentence, and keep it small: pseudocode for logic, a call tree for control flow, a `diff` sketch for what changes in an existing shape, a file tree for a new layout.

## Evidence

The exact command or test and its outcome. Strongest is a check that fails without the change and passes with it. When physics output changes (Veff, trigger rates, amplitudes, reference files), quantify by how much and on which data or test (see the `nuradio-tests` skill). A pure refactor shows unchanged output, e.g. the fixed-seed reference tests passing. List what was not exercised (other detectors, configurations, Python versions) so the reviewer knows where to look.

## Backwards compatibility & downstream impact

Does code, data or analysis outside the repo (user scripts, `.nur`/HDF5 files, detector descriptions, configs) keep working and keep giving the same answer after the merge?

- **API:** public functions, classes, signatures, defaults, module `begin`/`run` arguments. `extended` means old code keeps working; `breaking` names what callers must change (prefer deprecating, see `AGENTS.md`).
- **Data:** `.nur` and HDF5 output, detector JSON, YAML configs. Inputs that loaded before and now fail (or the reverse); outputs older versions can no longer read.
- **Results:** whether unchanged code on unchanged input now gives different numbers or trigger decisions, for which settings, by how much. A silent change in results deserves the most attention: it raises no error downstream.
- **Migration:** the concrete step for affected users (rename a call, set an option to restore the old behaviour, regenerate a file).
