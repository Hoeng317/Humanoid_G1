# Pilot exclusion record

This 500-update recovery pilot was intentionally interrupted after learning
iteration 134.  A source audit against `ManagerBasedRLEnv.step()` found that
auto-reset environments were reset to their sampled reference time and then
advanced by one 0.02 s command frame before the returned observation, even
though the new episode had not executed a physics step.

The run is excluded from checkpoint selection, validation, and final results.
The timing contract was corrected in `accad_g1/tracking_task.py`, covered by a
CPU regression test, and a fresh seed-42 pilot is required.  The original
`run_summary.json` remains `initializing` because SIGINT occurred inside the
simulator loop; this exclusion file is the authoritative reason it must never
be treated as a completed run.
