# PhyEditing evaluation

The accepted frozen pipeline uses SAM3.1 with integer image boundaries, visible-outline projection and the original anchor fitter. Inversion tuning is complete. Initial velocity is estimated jointly; no reference velocity is supplied.

Across all **1038 original single-view 30fps cases and 13918 observation frames**, mean relative gravity error is **2.704381%**, median **2.037109%**, p95 **7.473846%**, maximum **15.983240%**. Failure rate is **0%** and coverage **100%**. There are 1028/1038 cases within 10% and 1038/1038 within 20%. The user accepted this accuracy to preserve method simplicity; this is not a claim of maximum error below 10%. No subpixel refinement or further inversion tuning is planned.

These are inspected development data, not an untouched test. Background variants share physics groups and must not be counted as independent physical experiments. [Frozen aggregate evidence](sam31_frozen_result.json) includes event, gravity and camera statistics and source-result hashes; per-item research records remain local. The known-initial-velocity development protocol previously reached 1.11% mean / 6.2% maximum and is a separate information setting.

## Frozen comparison

All eight estimators receive the same declared camera, initial geometry/pose, angular motion, drag, gravity direction, motion constraints and actual initial prompts. All use the same frozen SAM3.1 boxes, timestamps and 13918 original observation frames; neural models read the same pixel-certified lossless cache. Reference initial velocity and target gravity are excluded from fitting inputs. The existing 1038-case development scope was fixed before this comparison; it does not measure eligibility across the whole dataset.

| Estimator | Successful fits | Mean error* | Maximum error* | Failure rate |
|---|---:|---:|---:|---:|
| GravTrace + SAM3.1 | 1038/1038 | 2.704% | 15.983% | 0.00% |
| Rotation-aware 2D | 1038/1038 | 5.297% | 36.364% | 0.00% |
| DepthPro | 696/1038 | 45.967% | 446.762% | 32.95% |
| DA-V2 Metric Indoor | 825/1038 | 48.951% | 687.830% | 20.52% |
| ZoeDepth | 771/1038 | 48.295% | 425.521% | 25.72% |
| MoGe-2 | 856/1038 | 49.926% | 703.529% | 17.53% |
| UniDepthV2 | 809/1038 | 59.556% | 1895.673% | 22.06% |
| VGGT | 1012/1038 | 43.860% | 684.525% | 2.50% |

*Mean/maximum use successful fits. Every failure retains the original 1038-case denominator; successful-fit coverage is one minus failure rate. All six depth networks produced complete depth outputs; their reported inversion failures are gravity-bound fits, not missing videos or inference errors.

The 2D control linearises outline translation at the declared initial position and compensates declared rotation. Depth baselines take median depth in the central half of each frozen box, lift its centre ray, anchor scale to declared initial surface depth and estimate unknown velocity plus gravity with the exact declared drag/time basis. Depth scale at release is extrapolated from all fitting depths, including fixed nonzero time offsets.

The lifting representation approximates object centres using surface depth and does not project the full rotating silhouette. It uses metric soft-L1 with scale 1m; GravTrace and 2D use pixel edge residuals. Their parameterisations/optimisers also differ. These are information-matched implemented estimators, not identical forward models or evidence that the neural networks themselves are universally inferior.

GravTrace's mean is 48.946% lower than the 2D control on all 1038 cases. Across 723 physics groups its equal-weight mean is 2.671729%; the 2D mean is 5.684200%. On each depth baseline's common-success subset, GravTrace also has a lower mean; these supplementary paired comparisons do not replace full-denominator failure reporting. [Comparison evidence](sam31_frozen_comparison.json) includes event/gravity/camera statistics, physics-group summaries, paired group resampling, checkpoint and input hashes, and the fixed protocol. Full per-case and per-group records remain local.

## Next evaluation

Validate physics configurations outside the metric-development groups without retuning, while recording prior exposure and respecting the [joint benchmark split](../joint_split_v3/PROTOCOL.md). Generator event holdouts do not automatically imply an unseen metric test. Then verify the SAM3.1 video-generation pipeline; training remains last. Current evidence does not establish independent generalisation or global optimality. Exploratory trials remain local.
