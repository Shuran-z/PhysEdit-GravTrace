# PhyEditing evaluation

The primary known-velocity development protocol reaches 1.11% mean / 6.2% maximum over 1038 successful fits. The centred-offset common-observation control reaches 0.89% / 8.99% over 1036/1038. These are separate protocols, not a chronological performance comparison.

Unknown-velocity default results are 2.88% / 19.83% at 30fps; 15fps has 845/1038 successful fits, mean 3.17%, maximum 34.16%. This scope does not yet meet the requested maximum error. No independent untouched physics-setting test has been completed.

Same-frame comparisons retain failed items and report event, gravity, camera and physics-group statistics. Six neural depth models and 2D controls have aligned retained observations and declared velocity bounds; geometry/spin/drag representation and residual-unit differences remain. Oracle-depth references receive extra truth information and cannot support a claim of global optimality.

## Evidence

- `common_observations.json`: known-velocity common observations.
- `offset_comparison.json`: offset and frame-rate controls.
- `bounded_2d.json`, `bounded_depth.json`: bounded unknown-velocity baselines.
- `vggt_common_result.json`: supersedes cached VGGT results after exact-input alignment.
- `sampling_audit15.json`: coverage limitations at 15fps.
- `generation_summary.json`: generation pipeline outputs; generated-video quality is not yet at target.

Trial logs and unsuccessful algorithm variants are archived locally; they do not change default predictions. Publication claims must preserve protocol, coverage and held-out limitations.
