# PhyEditing evaluation

The accepted frozen pipeline uses SAM3.1 with integer image boundaries, visible-outline projection and the original anchor fitter. Inversion tuning is complete. Initial velocity is estimated jointly; no reference velocity is supplied.

Across all **1038 original single-view 30fps cases and 13918 observation frames**, mean relative gravity error is **2.704381%**, median **2.037109%**, p95 **7.473846%**, maximum **15.983240%**. Failure rate is **0%** and coverage **100%**. There are 1028/1038 cases within 10% and 1038/1038 within 20%. The user accepted this accuracy to preserve method simplicity; this is not a claim of maximum error below 10%. No subpixel refinement or further inversion tuning is planned.

These are inspected development data, not an untouched test. Background variants share physics groups and must not be counted as independent physical experiments. [Frozen aggregate evidence](sam31_frozen_result.json) includes event, gravity and camera statistics and source-result hashes; per-item research records remain local. The known-initial-velocity development protocol previously reached 1.11% mean / 6.2% maximum and is a separate information setting.

## Next evaluation

Freeze the current predictions before comparing direct 2D inversion and DepthPro, DA-V2 Metric, ZoeDepth, MoGe-2, UniDepthV2 and VGGT. Give each method the same declared camera, geometry and initial prompts; unknown-velocity methods must not receive reference velocity. Align actual observation frames and decoded pixels, report any representation or loss differences, and retain all failed items in the original coverage denominator. Existing known-velocity comparisons do not substitute for this frozen unknown-velocity comparison. Oracle-depth references receive extra truth and must be labelled accordingly.

Report event, gravity, camera and physics-group summaries. Evaluate untouched physics settings without retuning, then verify the video-generation pipeline. Training remains last. Current evidence does not establish global optimality. Exploratory failures and trial logs remain local.
