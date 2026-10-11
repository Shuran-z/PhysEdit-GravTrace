# PhysEdit-GravTrace

Estimate gravity from a single-view video using **SAM3.1** object tracking and declared camera/object geometry. Current evaluation uses **PhyEditing**.

## Validated development results

| Protocol | Successful fits | Mean error | Maximum error |
|---|---:|---:|---:|
| Known initial velocity | 1038/1038 | 1.11% | 6.2% |
| Unknown initial velocity, 30 fps, frozen SAM3.1 pipeline | 1038/1038 | **2.704%** | **15.983%** |

The frozen single-view pipeline retains all 1038 cases and 13918 observation frames: failure rate **0%**, coverage **100%**. It uses integer image boundaries and the original anchor fitter. These are inspected development results; the protocols differ and do not establish independent generalization or global optimality. Inversion tuning is complete; the frozen comparison is complete, with independent configuration and generation validation next.

## Quick start

```bash
pip install -e .[test]
python -m gravtrace fit samples.jsonl -o predictions.jsonl --workers 8
python -m gravtrace score samples.jsonl predictions.jsonl -o summary.json
```

GravTrace projects a 3D flight trajectory and rotating object outline into the camera, clips the outline to the image, and jointly fits gravity and unknown motion parameters. Truth gravity is used only for scoring.

The [frozen comparison](docs/phyediting/validation.md#frozen-comparison) covers 2D and six depth estimators with matched information, frames and decoded pixels: 2D mean error is **5.297%**, versus **2.704%** for GravTrace.

- [Method and sample format](docs/REFERENCE.md)
- [Evaluation scope and frozen results](docs/phyediting/validation.md)
- [Video-generation pipeline](docs/PIPELINE.md)

MIT — see [LICENSE](LICENSE).
