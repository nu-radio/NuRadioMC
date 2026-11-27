<!--
Before opening, check (https://nu-radio.github.io/NuRadioMC/Introduction/pages/contributing.html):
- the PR targets `develop` and has at least one label (a component such as NuRadioMC / NuRadioReco / RNO-G plus a type such as Bug / New Feature / cleanup)
- new functions and classes have docstrings
- changelog.txt has a line under the current -dev version
Once approved, wait at least 24 hours after the last change before merging, so others can still comment.
-->

## TL;DR

<!-- Fragments fine. Blast radius: the verdict of the backwards compatibility section below, or "none". -->

- **Goal:** <!-- 1 sentence -->
- **Done:** <!-- max 2 sentences -->
- **Blast radius:** <!-- 1 sentence: API change, changed results or data, what users must do -->

## Summary

<!-- Detail beyond the TL;DR, without repeating it: context, the problem, how the fix works. Link the issue if there is one. Optional: a small diagram, diff sketch or call tree. -->

## Evidence

<!-- The test or check you ran and its result. If physics output changes (Veff, trigger rates, amplitudes, reference files): by how much, on which data or test. List what you did not test. -->

- 
- Not tested: 

## Backwards compatibility & downstream impact

<!-- Do user scripts, stored .nur/HDF5 files, detector descriptions and configs keep working and give the same answer after the merge? -->

- **API:** unchanged / extended / breaking — 
- **Data:** unchanged / changed — 
- **Results:** unchanged / changed — 
- **Migration:** <!-- what users do to adapt or keep the old behaviour; delete if nothing -->
