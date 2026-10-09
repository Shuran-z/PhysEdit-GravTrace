# GravTrace

Current work uses **PhyEditing**. The earlier PISA, NewtonBench and self-rendered evaluations below are historical references, not the acceptance set for this project.

The target is 2–3% mean relative gravity error, maximum ≤10% (20% is a fallback ceiling). An experimental all-frame offset estimate reaches 0.89% mean / 8.99% maximum on 1,036 common-observation items with known velocity; its unknown-velocity tail still misses the target, so it is not the default. **This is not yet met across all settings.** The tuned, filtered set with known launch velocity reaches 1.11% mean / 6.2% maximum; with unknown launch velocity it reaches 2.88% / 19.8% at 30 fps, and 3.17% / 34.2% among 845 successful fits out of 1,038 at 15 fps.

See [validation notes](docs/phyediting/validation.md), [interactive report](docs/phyediting/report.html) and [six-model results](docs/phyediting/generation_summary.json).

GravTrace measures the gravity a video shows. Given a single-view video of an object that falls,
is thrown, or slides down a ramp, together with what is known about the scene before the motion
starts (camera, the object's shape and initial pose, the direction of gravity, any declared
initial velocity), it recovers the gravitational acceleration g, in m/s², that best explains the
object's image motion. It is the gravity-inversion metric of the PhysEdit benchmark: a physically
faithful video should imply the gravity it was asked to show.

## How it works

1. **Observe.** Each frame's target mask becomes an image box.
2. **Predict.** Points on the object's outline (its 8 bounding-box corners, or the hull of its mesh)
   are carried along an analytic trajectory, projected with the known camera, and the image box they
   span is taken. Trajectories: free flight, with a declared linear drag; sliding or rolling on a
   slope with friction, including leaving a ramp over its top edge into free flight. The object keeps
   its initial orientation unless the sample declares its spin: then the outline turns by torque-free
   rigid-body rotation (Euler's equations with the declared inertia and angular damping), which does
   not depend on g.
3. **Compare.** The residual is the displacement of each box edge since an anchor frame, predicted
   minus observed. Displacements cancel any constant offset between the declared geometry and the
   mask (pivot, mesh scale, mask bias); an edge on the image border, or one the sample declares hidden
   behind another object, is ignored.
4. **Fit.** Robust (soft-L1) bounded least squares over log g and whatever the sample leaves open,
   each inside its declared range: the time of the first frame after the initial state, the launch
   speed, elevation and azimuth of a projectile, or the along-slope speed and distance to the ramp
   top. Several starting points are tried; discrete hypotheses (ramp orientation) compete by cost.
5. **Stop at contact.** Analytic motion ends at the first contact, so the fit window ends there: at
   a predicted crossing of the declared floor, or at a frame that the flight fitted to the
   preceding frames misses by more than max(3 px, 3 × its median frame error).

An experimental `offset_mode: centred` estimates constant per-edge offsets from all usable frames; the default remains `anchor`. `window.contact_check: false` disables adaptive contact trimming only for fixed-window controls with verified contact-free input.

Each prediction reports g, a status, the frames used, the fitted nuisances, and `sensitivity_px`:
how much the image motion changes between g/2 and 2g. A few pixels or less means the video barely
constrains g, and such estimates should not be trusted (see the validation below).

## Sample format

One JSON object per line. `truth` is read only by `score`.

| field | content |
|---|---|
| `id`, `scenario` | `freefall`, `projectile` or `incline` |
| `fps`, `image_size` | frame rate; `[width, height]` in pixels |
| `camera` | `matrix_world` (4×4 camera-to-world, camera looks along −Z), `fx`, `fy` (focal lengths in units of image width and height), optional `cx`, `cy` (principal point, default 0.5) |
| `object` | `corners` (outline points in the object frame, metres, scale applied: 8 box corners or a mesh hull; omit if unknown), `position` (m), `quaternion` (xyzw); optional spin: `angular_velocity` (rad/s, world), `inertia` (principal moments, kg m², object frame), `mass` (kg), `angular_damping` (N m s), `linear_damping` (N s/m) |
| `features` | `box` (default) or `centre` (fit the box centre only) |
| `motion.gravity_dir` | unit vector, default `[0, 0, -1]` |
| `motion.t0` | `[lo, hi]` time (s) of the first fitted frame after the declared initial state |
| `motion.v0` | known initial velocity (m/s); freefall defaults to rest |
| `motion.speed`, `motion.angle_deg`, `motion.directions` | projectile with unknown launch: speed range, elevation range, optional azimuth seeds |
| `motion.slopes`, `motion.speed` | incline: list of `{dir, angle_deg, mu, rolling, length}` hypotheses (`dir` points downslope, `length` is the ramp length) and the along-slope speed range (negative is upslope) |
| `motion.ground_z` | height of the floor the object can land on (optional) |
| `window` | `start_frame` (first frame of motion), `max_frames` (default 6 / 8 / 16 by scenario), optional `edges_visible` (frame → `[left, top, right, bottom]` booleans: edges hidden behind other declared objects) |
| `masks` | `dir`, `pattern` (e.g. `mask_*.png`, `segmentation_*.npy`), `label` (id in a label map, or null for binary masks) |
| `boxes` | instead of `masks`, boxes from any tracker: `frames`, `xyxy` (pixel edges), optional `times` (s) |
| `truth.gravity` | m/s² |

## Usage

```bash
pip install -e .[test]
python -m gravtrace fit samples.jsonl -o predictions.jsonl --workers 16
python -m gravtrace score samples.jsonl predictions.jsonl -o summary.json
pytest tests
```

`fit` removes `truth` before any sample reaches a worker, and a test checks that the fitting code
never reads it. `score` reports error statistics per source and scenario, and the
coverage-penalised mean used as the benchmark score: min(relative error, 1) averaged over all
samples, where a failed or missing fit counts as 1. `--root OLD=NEW` rewrites mask paths when the
data live somewhere else.

## Scoring generated videos

A generated video continues a ground-truth scene from a conditioning frame, so it is scored against
that scene's declaration. `scripts/generated_manifest.py GT.jsonl VIDEOS.jsonl OUT.jsonl` copies
each video's ground-truth sample and replaces only the observation: the video's masks, its frame
rate, and the time of its first frame (`VIDEOS.jsonl` rows: `id`, `gt_id`, `masks`, `fps`,
`start_s`). List every generated video, including those whose tracking failed; they score as
misses. Then run `fit` and `score` as above.

Compare models against the real videos passed through the same mask pipeline. With SAM2 masks on
the PISA and NewtonBench release videos that floor is 7.7 % (PISA 2.1 %, NewtonBench 11.1 %),
while nine current video models score 79–97 %. For the self-rendered scenes, whose free flights
are short, the floor depends on the frame rate: 6.0 % at 30 fps, 8.5 % at 24, 26 % at 15–16 and
60 % at 8 fps. Rank models against the floor at their own frame rate.

## Historical validation (superseded dataset)

Relative gravity error on the 260 rendered ground-truth videos with known object geometry:

| source | n | mean | median | max | ≤ 30 % |
|---|---:|---:|---:|---:|---:|
| PISA (freefall) | 60 | 1.9 % | 1.7 % | 8.1 % | 60 |
| NewtonBench | 100 | 7.2 % | 5.0 % | 55 % | 98 |
| self-rendered | 100 | 6.7 % | 4.6 % | 40 % | 97 |
| **all** | **260** | **5.8 %** | **3.1 %** | 55 % | 255 |

By scenario: freefall 3.1 % (n = 94), projectile 6.4 % (n = 126), incline 10.1 % (n = 40).
Error tracks `sensitivity_px`: 17.9 % mean below 3 px (n = 12), 8.9 % at 3–10 px (n = 30),
4.7 % above 10 px (n = 218).

## PhyEditing

PhyEditing (private Hugging Face dataset `phyeditingvideo/Phyediting`) renders rigid-body table-top
scenes with Genesis at five gravities (1.62, 3.71, 9.81, 15 and 24 m/s²), three synchronised cameras,
30 fps and 210 frames, with the simulator states of every object in every frame. Most of its motion
is pushing, sliding and toppling; gravity shows cleanly where an object flies free (off a table edge,
a shelf, a stack), and those flights are where it is measured.

### Preparing the data

```bash
python scripts/phyediting_extract.py data/phyediting data/compact          # every release -> compact records
python scripts/phyediting_calibrate.py data/compact --prefix T08V4F480 --apply  # measured image offsets
python scripts/phyediting_windows.py data/compact runs/windows.jsonl        # free-flight windows
python scripts/phyediting_samples.py data/compact runs/windows.jsonl runs/oracle_cam01.jsonl \
    --camera cam01 --min-frames 4 --jobs runs/jobs_cam01.jsonl --video-root DATA   # samples + SAM2 jobs
python scripts/track_sam2.py runs/jobs_cam01.jsonl runs/tracks.jsonl --sam2 SAM2 --ckpt sam2.1_hiera_small.pt
python scripts/phyediting_samples.py ... --tracks runs/tracks.jsonl          # samples observed by SAM2
python scripts/phyediting_benchmark.py runs/benchmark --oracle ... --oracle-pred ... --observed ...
```

- **Records.** The releases use different layouts; `phyediting_extract.py` reads them all. A camera's
  view comes from that camera's render record. `configs/` files can be stale (T06, T08, T09 and T10
  differ from what was rendered), so a camera known only from configs is used only when the same
  trajectory's cam01 config matches its render record.
- **Calibration.** Every T08 video sits 36–43 px lower than its recorded camera predicts (all
  backgrounds and cameras, within 1.5 px per camera). `phyediting_calibrate.py` measures the shift that
  aligns the projected moving objects with the pixels that change, and records it as a principal-point
  offset. Other releases measure 0 px.
- **Windows.** A flight is a run of frames whose acceleration equals the gravity vector (no contact
  force). Its declared initial state is the simulator's at the first frame: position, orientation,
  velocity and spin, with the collision proxy's inertia and the declared damping (Genesis damps the
  angular velocity of free bodies, e.g. by 0.018 N m s on a book). Outlines are the visual meshes' hulls
  where a copy is available, else the proxies'.
- **Visibility.** Rays from the camera to the object's surface are tested against every other declared
  object, giving for each frame which box edges are visible; hidden edges drop out of the fit.
- **Items.** Per video, the window that best constrains g, provided it is observable and trackable:
  in view in ≥ 4 frames with ≥ 2 visible edges, GravTrace's `sensitivity_px` on the true boxes ≥ 20 px,
  and SAM2 (box prompt in the condition frame) following the object in the ground-truth video (no
  visible edge more than 4 px off the projected true box once a constant offset is removed).
- **Duplicates.** 11,522 extracted trajectories hold 6,922 distinct motions: background-only variants
  (bg02/bg03/bg04, sometimes bg01), deterministic repeats and static trajectories share a physics group.

### Ground-truth floor

1,038 videos (events T03, T08, T17, T18, T19, T20; all five gravities; three cameras), SAM2 masks:

| method | mean | median | p95 | max | failed |
|---|---:|---:|---:|---:|---:|
| GravTrace | 1.11 % | 0.83 % | 3.2 % | 6.2 % | 0 |
| GravTrace, reference-bounded launch velocity (`free`) | 2.88 % | 2.20 % | 8.1 % | 19.1 % | 0 |
| GravTrace, box centre only | 1.37 % | 1.01 % | 3.7 % | 29.3 % | 0 |
| GravTrace, no spin model | 2.41 % | 1.14 % | 9.9 % | 36.1 % | 0 |
| 2D parabola, scale from object size | 16.92 % | 14.13 % | 50.3 % | 165.1 % | 2 |
| 2D parabola, scale from declared depth | 11.95 % | 7.18 % | 47.6 % | 199.9 % | 2 |
| 2D, same information as GravTrace | 3.69 % | 2.11 % | 17.2 % | 37.1 % | 2 |
| lifted with true depth | 5.11 % | 1.99 % | 22.0 % | 112.0 % | 2 |
| lifted with true depth, declared velocity | 1.91 % | 0.67 % | 7.3 % | 55.7 % | 2 |

By gravity (n, mean, max): 1.62: 512, 1.06 %, 6.2 %; 3.71: 205, 1.01 %, 5.2 %; 9.81: 140, 1.43 %,
5.9 %; 15: 75, 0.90 %, 3.7 %; 24: 106, 1.31 %, 5.4 %. By camera: cam01 725, 1.01 %, 6.2 %; cam02 87,
1.34 %, 6.1 %; cam03 226, 1.36 %, 5.9 %. With a 15 px sensitivity gate, 1,351 videos: mean 1.41 %,
max 11.6 %.

These are development-set results: the sensitivity gate was chosen after examining this batch. They do not establish whole-dataset generalization or global optimality. A future independent evaluation must freeze the protocol and split by motion/physics groups before tuning; splitting this already-inspected batch now would not make it an untouched test set. Means exclude failed fits, whose counts must be reported alongside them.

The baselines receive the same windows, tracks and declared visibility, but currently drop frames with any hidden edge while GravTrace can use partially visible frames. The stricter comparison on identical, fully visible and unclipped frames is reported below. "Same information" means the camera, the
declared position and velocity and the gravity direction, with the image track modelled as the declared
flight under the projection linearised at the start; the lifted baselines back-project the box centre
with the known intrinsics and fit g along the known gravity direction. `scripts/baselines.py` computes
them; `scripts/depth_probe.py` adds monocular and multi-view depth models (DepthPro, Depth Anything V2
metric, ZoeDepth, MoGe-2, UniDepthV2, VGGT) in the same way, raw or with their scale anchored to the
declared initial state; `scripts/phyediting_report.py` prints the tables.

### Comparison on identical observations

All methods receive the same fully visible, unclipped frames; GravTrace contact truncation is disabled for this comparison. 1,036 / 1,038 items have at least four common frames. This is still a tuned development set. Prediction-depth methods are anchored to the declared depth and given the declared velocity. Oracle depth uses simulator truth and is a reference ceiling, not a deployable competitor.

| method | mean error, successful fits | maximum | failed or missing / 1036 | coverage-penalised score |
|---|---:|---:|---:|---:|
| ours_centred_experimental | 0.89% | 8.99% | 0 | 0.89% |
| ours | 1.19% | 15.62% | 0 | 1.19% |
| lift_depthpro_anchored_v0 | 35.09% | 521.90% | 49 | 34.30% |
| lift_moge2_anchored_v0 | 32.64% | 915.17% | 34 | 28.89% |
| lift_unidepth_v2_anchored_v0 | 33.92% | 1457.29% | 20 | 26.94% |
| pixel2d_matched | 3.10% | 57.27% | 0 | 3.10% |
| lift_zoedepth_anchored_v0 | 29.22% | 949.76% | 30 | 29.08% |
| lift_dav2_metric_anchored_v0 | 30.00% | 751.47% | 11 | 24.48% |
| lift_oracle_v0 | 0.87% | 16.68% | 0 | 0.87% |
| lift_vggt_anchored_v0 | 20.44% | 169.56% | 39 | 23.27% |

The oracle-depth reference has a lower mean (0.87%) than the default GravTrace (1.19%) and the experimental centred variant (0.89%). GravTrace is better on mean error than the tested 2D and predicted-depth methods; this does not establish global optimality. The common-frame maximum of 15.62% still exceeds the 10% target. Coverage-penalised score averages min(relative error, 1), with failed/missing fits assigned 1.

Reproduce with `phyediting_common_observations.py`, `baselines.py`, `gravtrace fit`, then `phyediting_common_report.py`. Source: [common_observations.json](docs/phyediting/common_observations.json).

### Full unknown-launch offset experiment

Both modes retain the same 1,038 development items and attempt the same reference windows. No high-error item is removed. `centred` estimates the fixed per-edge offset across usable frames instead of subtracting one anchor; it remains experimental.

| fps | mode | successful / attempted | mean error | maximum | within 20%, out of all 1038 |
|---|---|---:|---:|---:|---:|
| 30 | anchor (default) | 1038 / 1038 | 2.88% | 19.83% | 1038 |
| 30 | centred | 1038 / 1038 | 2.64% | 21.06% | 1037 |
| 15 | anchor (default) | 845 / 1038 | 3.17% | 34.16% | 843 |
| 15 | centred | 845 / 1038 | 2.93% | 28.34% | 844 |

The centred variant improves the mean at both frame rates and the 15 fps maximum, but worsens the 30 fps maximum. Neither variant satisfies maximum ≤10% across unknown-launch settings. The 193 low-frame-rate failures remain in the denominator. See [per-event, per-gravity and physics-group statistics](docs/phyediting/offset_comparison.json). Reproduce with `phyediting_offset_trial.py --fps 30 --limit 0 --all-items --mode centred` and its 15 fps equivalent, then `phyediting_offset_comparison.py`.

### Generated videos (gravity editing)

The first 24 frames of every PhyEditing video are static, so frame 23 is the same at all five
gravities: it is the condition frame, and the prompt describes the event and the target gravity
(`scripts/phyediting_gen.py build`). A generated video is tracked from its first frame with the
declared boxes (`phyediting_gen.py jobs`); its camera motion is estimated per frame as a similarity onto
frame 0 (ORB features, RANSAC; `scripts/camera_motion.py`) and removed from the boxes, which are then
fitted in the item's window, timed by the model's frame rate (`phyediting_gen.py score --camera-motion`).

Generated-video scoring defaults to `--v0 agnostic`: launch speed (0–8 m/s) and direction are fitted with g,
and only the declared position at the window start (for depth) and the window times come from the
reference trajectory. On the ground-truth videos this scores 2.88 % mean, 19.8 % max (none above 20 %);
resampled to 15 fps, 845 / 1,038 windows fit, with 3.17 % mean and 34.2 % maximum error. Position, orientation, spin and window timing still come from the reference, so this is not a fully video-only estimator. The declared-velocity protocol of the floor above
(`--v0 declared`: 1.11 % mean, 6.2 % max) must not be used for generated videos, because the reference
velocity at the window start carries the target gravity. `scripts/phyediting_gen_nulls.py` shows this with
two synthetic generators scored like a model:

| generator | `--v0 declared` | `--v0 agnostic` |
|---|---|---|
| object never moves | 11.8, 17.3, 32.8 m/s² at targets 9.81, 15, 24 | median near the 0.1 search bound; inspect individual records |
| always Earth gravity (same scene at 9.81) | 2.78 and 13.5 at targets 3.71 and 15 | 10.1 at target 9.81, 0.1 elsewhere (landed) |

`--v0 free` (speed within ±50 % of the reference, any direction) is in between: it reads the bound for the
static generator but still takes the speed range from the reference.

### Six-model generation test

Each model has 40 scoring records. Fixed-window `agnostic` results are shown below; fits at a search bound are diagnostic, not reliable gravity measurements. Missing tracks remain in the 40-video success-rate denominator.

| model | fitted, including bounds | at bound | no track | relative error ≤20%, out of 40 |
|---|---:|---:|---:|---:|
| LTX-Video-2B | 37 | 23 | 3 | 0 |
| Wan2.2-TI2V-5B | 31 | 23 | 9 | 0 |
| CogVideoX1.5-5B | 30 | 15 | 10 | 0 |
| Cosmos3-Nano | 35 | 25 | 5 | 0 |
| Wan2.2-I2V-A14B | 39 | 23 | 1 | 2 |
| PhysAlign (Wan2.2-A14B) | 38 | 16 | 2 | 1 |

The pipeline produced 240 scoring records, but these results do not demonstrate successful gravity editing. Whole-clip window scans are a lenient multiple-attempt diagnostic, not the primary success rate; target substitution is not a significance test.

### Remaining work

1. Tune unknown-launch estimation and low-frame-rate tail error, retaining failure counts and group coverage.
2. Extend the completed common-observation comparison to unknown-launch settings and audit remaining differences in geometric/spin information.
3. Freeze the protocol and evaluate unseen physics setups, grouping background variants and synchronized cameras together.
4. Expand generation tests after evaluator controls pass; start training only after the acceptance criteria and evaluation protocol are stable.

## Assumptions and limitations

- **Rigid object and declared spin.** Free-flight spin is modelled when angular velocity, inertia and damping are supplied. Unmodelled rotation, deformation and contact remain sources of error.
- **No collisions.** Motion is analytic; the fit window ends at the first contact.
- **Time.** Frame times are frame index / fps, measured from the declared initial state. A video that
  plays in slow motion implies a smaller g.
- **Frame rate.** Short free flights need enough frames. The self-rendered scenes reach a contact
  about 0.15 s after release; their ground-truth videos resampled to lower rates give a median error
  of 4.6 % at 30 fps, 3.3 % at 24, 13 % at 16, 15 % at 15 and 68 % at 8 fps. Compare generated videos
  against a control at the same frame rate, and do not score g from 8 fps clips of such scenes.
- **Declared initial state.** Camera, geometry, initial pose and the declared velocity or its range
  come from the scene description; if the video's own first frame differs from that state (for
  example a different launch speed), the difference is absorbed into g.
- **Development set.** The window lengths, the contact rule and the loss scale were chosen on the
  same 260 videos reported above, so those numbers are development-set error. The self-rendered
  100 were themselves curated from a larger pool.
- **Missing geometry.** A sample without `corners` falls back to the box centre (9 of the 260,
  whose meshes were unavailable).
- WorldBench is not covered yet: its object meshes and initial poses are not available.

## Generating and evaluating models: `physedit`

The repository also holds the driver that runs the PhysEdit benchmark end to end for any set of registered video
models: it builds each model's manifests, generates on free GPUs across the machines it knows, collects the videos,
runs the evaluation stages (V/P metrics, Qwen judge, SAM2 masks, GravTrace) and rebuilds the table.

```bash
python3 -m physedit run public211 --models ltx_i2v,cosmos3_nano --watch 10
python3 -m physedit status public211
```

Models, machines and benchmarks are declared in `physedit/config.py`; see [docs/PIPELINE.md](docs/PIPELINE.md).

## License

MIT, see [LICENSE](LICENSE).

## Converting the earlier manifests

`scripts/convert_v22.py` turns a GravTrace v22 manifest (PISA, NewtonBench and the self-rendered
scenes) into this format, resolving the target from the declared segmentation label and reading
object geometry from Google Scanned Objects or the primitive's dimensions.

### Experimental subpixel SAM2 boxes

`python scripts/track_sam2.py --help` exposes `--box-mode pixel|subpixel|both`.
The default remains `pixel`. `both` extracts integer boxes and interpolated zero-crossing
boxes from the **same logits**, saving `tracks` and `tracks_subpixel` for paired analysis.
A ten-video development pilot (five error-selected cases, five fixed-hash controls)
did not meet the unknown-velocity tail target: subpixel maxima were 20.49% at 30 fps
and 21.93% at 15 fps, with coverage 10/10 and 8/10 respectively.
This is diagnostic evidence, not a benchmark-wide improvement or held-out result.
See [paired results](docs/phyediting/subpixel_pilot.json) and
[work status](docs/phyediting/WORK_STATUS.md).

Edge-level development diagnostics are available in [edge_diagnosis.json](docs/phyediting/edge_diagnosis.json). They separate constant/linear bias from quadratic observation error at the original timestamps. Simulator geometry is used only for diagnosis, never to correct fitted observations.

Fixed-axis ablation on the ten-video development pilot did not improve the tail; the default still uses all visible edges. All ablation modes disable contact trimming to isolate the axis change. See [edge_ablation.json](docs/phyediting/edge_ablation.json). The full 15 fps contact-off control (`phyediting_offset_trial.py --no-contact-check`) completed: 845/1038 successful fits, mean 2.917%, maximum 24.654%. The contact-on control is 2.928% / 28.344% with identical coverage. This limited improvement does not meet the tail target; defaults are unchanged. See [contact comparison](docs/phyediting/contact_comparison15.json).

The [15 fps identifiability diagnostic](docs/phyediting/identifiability15.json) projects the fitted log-gravity Jacobian off unknown-velocity nuisance directions. Median retained local information is 5.38% among 845 successful fits; all 193 insufficient-frame cases remain listed. This is an uncalibrated local linear diagnostic, not an uncertainty guarantee or a new rejection rule.

A fixed-gravity nuisance profile on 14 error-selected 15 fps cases plus 5 fixed-hash controls found no materially lower grid cost than the saved estimates (largest relative cost improvement 6.39e-12). The grid and nuisance starts are finite; this is development diagnosis, not a global-optimality proof. See [profiles](docs/phyediting/gravity_profile15.json) and [figure](docs/phyediting/gravity_profile15.svg).

Experimental shared-gravity fitting has concurrent auxiliary observations in 269/1038 development videos. On those 269, mean error changes from 2.567% to 2.426%, but maximum increases from 9.686% to 14.682%. Fixed availability-based routing across all 1038 preserves 845 successes and a 24.654% maximum; defaults are unchanged. Additional object information is not matched to earlier single-object baselines. See [coverage](docs/phyediting/joint_coverage15.json), [full fits](docs/phyediting/joint_full15.json), and [group comparison](docs/phyediting/joint_comparison15.json).

A same-frame median-background motion-box development trial also failed to improve the tail: maximum error rises to 53.013% on the 19 diagnostic cases, and fixed-hash controls worsen. It is not enabled by default. See [negative trial](docs/phyediting/motion_refine15.json); no target-error-based fallback is applied.

Unknown-velocity common-fit-frame controls (30 fps, development data): ours 2.909% mean / 26.482% max, 2D 5.718% / 41.949%, true-depth reference 2.260% / 25.924%; each succeeds on 1036/1038. Six predicted-depth anchored methods average 43.010–61.263% among successes, with failures retained. Depth methods fail rather than fit a shorter sequence when any retained depth is missing; scale is anchored by extrapolating model depth to the declared start. Free-velocity priors still differ, and cached VGGT inference may include extra input frames: this is not the final fully matched comparison. See [unknown common controls](docs/phyediting/unknown_common.json).

A bounded 2D control now uses speed 0–8 m/s, elevation ±89 degrees and gravity 0.1–80, matching our declared bounds. It succeeds on 1036/1038, with mean 5.863% and maximum 39.059%. The linear projection/centre model still differs from our geometry/spin/drag model; depth-prior and VGGT neural-input alignment remain pending. See [bounded 2D groups](docs/phyediting/bounded_2d.json).

Six lifted-depth controls now use the same velocity/elevation/gravity bounds and require every retained depth frame. Successful means remain 42.860–62.377%, with 34–302 failures per 1038 retained in coverage. See [bounded depth groups](docs/phyediting/bounded_depth.json). Metric loss units and geometry/spin/drag representations still differ; VGGT neural input-frame alignment remains pending.

VGGT neural input-frame alignment is now complete: 502 windows rerun and 534 exactly matching inputs reused. After validating all 1036 ordered fitting-frame lists, the bounded unknown-velocity lift succeeds on 981/1038 (94.51%), with 54 boundary failures, one invalid depth sequence and two missing common windows. Successful mean/max errors are 46.881%/576.514%. This supersedes the cached VGGT control for this protocol; see [exact-input results](docs/phyediting/vggt_common_result.json). Geometry/spin/drag and residual-unit differences remain; these are development results, not a global optimum or independent validation.

All five single-image caches match retained frames, boxes, intrinsics and video paths (1036/1036 each); extra images are inferred independently. See [input audit](docs/phyediting/single_depth_input_audit.json). At 15fps, 191/1038 declared windows have capacity below four real frames and two more lack tracked frames. All 193 remain in the denominator; interpolation cannot add independent observations. See [sampling audit](docs/phyediting/sampling_audit15.json).

A disjoint-window same-object audit finds additional declared flights for 502/1038 items, but only 8/191 cadence-limited items and 4/14 existing >10% tails. A diagnostic pilot (those four tails plus eight fixed-hash controls) worsens mean/max from 5.804%/14.428% to 8.217%/60.905%; both fits return ok on all 12. This selected development experiment is not overall or unseen performance, and the default is unchanged. See [coverage](docs/phyediting/cross_window_coverage15.json) and [paired pilot](docs/phyediting/cross_window_pilot15.json).

The same-time oracle-box diagnostic on the existing 12-case cross-flight pilot gives joint maximum error 0.174108% and independent-window maximum 0.822801%. Declared start position, rotation and angular velocity exactly match compact metadata. The worst observed joint case has 63.341% auxiliary-only SAM2 error, versus 0.8228% with projected boxes. This supports an observation-error diagnosis; oracle controls do not improve official predictions or constitute held-out validation. See [diagnostic](docs/phyediting/cross_window_diagnosis15.json).

A fixed cross-window residual-noise scaling control retains all observations and estimates each scale from the independent unknown-velocity fit, with a 1px floor. On the existing selected 12-case pilot, weighted mean/max errors are 7.563%/51.052%, still worse than the single-window 5.804%/14.428%. Smoothness alone also fails to identify biased tracks; no quality threshold or error-based rejection was adopted. See [edge diagnosis](docs/phyediting/cross_window_edges15.json) and [weighted control](docs/phyediting/cross_window_weighted15.json).
