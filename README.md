# PhysEdit-GravTrace

Recover gravitational acceleration from tracked object motion in a single-view video, using declared camera and object geometry. Current evaluation uses **PhyEditing**.

## Validated development results

| Protocol | Successful fits | Mean error | Maximum error |
|---|---:|---:|---:|
| Known initial velocity, tuned filtered set | 1038/1038 | 1.11% | 6.2% |
| Known velocity, common observations, experimental centred offsets | 1036/1038 | 0.89% | 8.99% |
| Unknown initial velocity, default, 30 fps | 1038/1038 | 2.88% | 19.83% |

These are different evaluation protocols and inspected development sets, not independent test results. Means cover successful fits; failed items remain in coverage denominators. The known-velocity protocol meets the target. Unknown velocity and low frame rates still require work. Default offsets remain `anchor`.

## Quick start

```bash
pip install -e .[test]
python -m gravtrace fit samples.jsonl -o predictions.jsonl --workers 8
python -m gravtrace score samples.jsonl predictions.jsonl -o summary.json
pytest tests
```

Samples declare camera, object geometry/pose, gravity direction, motion window and tracked boxes or masks. Initial velocity can be declared or jointly estimated. Ground-truth gravity is used only for scoring.

## Method and evaluation

GravTrace projects an analytic 3D trajectory and rotating object outline into the camera, then fits gravity and unknown motion parameters with robust bounded least squares. Hidden and clipped edges are masked.

Comparisons include 2D inversion and DepthPro, DA-V2 Metric, ZoeDepth, MoGe-2, UniDepthV2 and VGGT. Retained observation frames and velocity bounds are aligned; forward-model representations and loss units still differ. Current evidence does not establish global optimality.

- [Method, sample format and detailed usage](docs/REFERENCE.md)
- [Evaluation scope and reproducible results](docs/phyediting/validation.md)
- [Video-generation pipeline](docs/PIPELINE.md)

Exploratory trials and research logs are kept locally. Only validated results and maintained interfaces belong in the main documentation.

## License

MIT — see [LICENSE](LICENSE).
