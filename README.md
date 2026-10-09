# GravTrace

GravTrace measures the gravity a video shows. Given a single-view video of an object that falls,
is thrown, or slides down a ramp, together with what is known about the scene before the motion
starts (camera, the object's shape and initial pose, the direction of gravity, any declared
initial velocity), it recovers the gravitational acceleration g, in m/s², that best explains the
object's image motion. It is the gravity-inversion metric of the PhysEdit benchmark: a physically
faithful video should imply the gravity it was asked to show.

## How it works

1. **Observe.** Each frame's target mask becomes an image box.
2. **Predict.** The 8 corners of the object's bounding box are carried along an analytic
   trajectory, projected with the known camera, and the image box they span is taken. Trajectories:
   free flight; sliding or rolling on a slope with friction, including leaving a ramp over its top
   edge into free flight. The object keeps its initial orientation.
3. **Compare.** The residual is the displacement of each box edge since an anchor frame, predicted
   minus observed. Displacements cancel any constant offset between the declared geometry and the
   mask (pivot, mesh scale, mask bias); an edge on the image border is ignored.
4. **Fit.** Robust (soft-L1) bounded least squares over log g and whatever the sample leaves open,
   each inside its declared range: the time of the first frame after the initial state, the launch
   speed, elevation and azimuth of a projectile, or the along-slope speed and distance to the ramp
   top. Several starting points are tried; discrete hypotheses (ramp orientation) compete by cost.
5. **Stop at contact.** Analytic motion ends at the first contact, so the fit window ends there: at
   a predicted crossing of the declared floor, or at a frame that the flight fitted to the
   preceding frames misses by more than max(3 px, 3 × its median frame error).

Each prediction reports g, a status, the frames used, the fitted nuisances, and `sensitivity_px`:
how much the image motion changes between g/2 and 2g. A few pixels or less means the video barely
constrains g, and such estimates should not be trusted (see the validation below).

## Sample format

One JSON object per line. `truth` is read only by `score`.

| field | content |
|---|---|
| `id`, `scenario` | `freefall`, `projectile` or `incline` |
| `fps`, `image_size` | frame rate; `[width, height]` in pixels |
| `camera` | `matrix_world` (4×4 camera-to-world, camera looks along −Z), `fx`, `fy` (focal lengths in units of image width and height) |
| `object` | `corners` (8×3 bounding-box corners in the object frame, metres, scale applied; omit if unknown), `position` (m), `quaternion` (xyzw) |
| `motion.gravity_dir` | unit vector, default `[0, 0, -1]` |
| `motion.t0` | `[lo, hi]` time (s) of the first fitted frame after the declared initial state |
| `motion.v0` | known initial velocity (m/s); freefall defaults to rest |
| `motion.speed`, `motion.angle_deg`, `motion.directions` | projectile with unknown launch: speed range, elevation range, optional azimuth seeds |
| `motion.slopes`, `motion.speed` | incline: list of `{dir, angle_deg, mu, rolling, length}` hypotheses (`dir` points downslope, `length` is the ramp length) and the along-slope speed range (negative is upslope) |
| `motion.ground_z` | height of the floor the object can land on (optional) |
| `window` | `start_frame` (first frame of motion), `max_frames` (default 6 / 8 / 16 by scenario) |
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

## Validation on ground-truth videos

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

## Assumptions and limitations

- **Rigid, non-rotating object.** A tumbling object changes its box in ways the model does not
  capture; tumbling objects on inclines produce the largest errors above.
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
