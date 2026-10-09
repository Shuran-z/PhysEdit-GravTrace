# One-command generation and evaluation (design draft)

Goal: generate and evaluate any set of video models on a PhysEdit benchmark with one command,

```bash
python -m physedit run --benchmark public211 --models ltx_i2v,cosmos3_nano   # or --models all
python -m physedit status --benchmark public211                              # per model and stage, with ETA
```

replacing the per-group launch scripts used so far (hupanwen launch/scheduler scripts, wangzijun
`run_gpu_job.sh`, A800 `run_node.sh`, the hupanwen evaluation orchestrator, and the WSL sync and judge
daemons). The `gravtrace` package stays a standalone metric library; `physedit` is the driver.

## Pieces

| piece | content |
|---|---|
| benchmark | samples, prompts (coarse / fine), conditioning frames, ground truth. **Data layout follows the PhysEditing dataset on the Hub (`phyeditingvideo/Phyediting`)**; an adapter turns it into canonical rows. |
| model registry | per model: runner (diffusers, official repo, LoRA), parameters (steps, guidance, frames, fps, size), GPUs needed (H3: 2), conditioning mode (i2v frame, v2v prefix), weights and environment per host |
| host registry | per host: how to reach it (ssh alias, through WSL for the NSCC proxy), GPUs and which ones may be used, scratch and data roots, Python environments |
| runs | `x5` (V/P conditioning, aligned with self100) and `x1` (first-visible-frame conditioning for inversion) |
| stages | `manifests -> generate -> sync -> vp (SAM2, 1-s metrics, judge clips) -> judge (Qwen-VL) -> inversion (SAM2 masks + gravtrace) -> table` |

## Behaviour

- A model is placed on a host that has its weights and enough free GPUs (`--hosts` overrides). One model per GPU,
  shards over several GPUs when available.
- Every stage is idempotent: finished rows are skipped, so `run` can be repeated after any interruption, and a
  `--watch` mode advances each model to its next stage as soon as the previous one completes.
- Remote jobs run detached (tmux / setsid) and write status files the driver reads for `status`.
- `--dry-run` prints the placement and every command without running anything.

## Existing pieces it reuses

| stage | script today |
|---|---|
| manifests | hupanwen `make_eval_manifests.py`, `build_inv_x1.py` |
| generate | hupanwen `run_diffusers_i2v_batch_strict_perrow.py`, `run_worldbench_cosmos_batch.py`; Hunyuan `run_diffusers_perrow.py`; wangzijun `run_dynamicrafter.py`, `run_physrvg.py`, VideoGPA script + `finalize_videogpa.py`; A800 `run_single_a800_manifest.py`, `run_cosmos3_perrow.py`, `run_h3_perrow.py` |
| sync | WSL `sync_to_hupanwen.sh` |
| vp | hupanwen `vp_port/build_jobs_public211.py`, `run_repair_public211.py`, `score_window_1s_public211.py`, `build_qwen_proxy_public211.py` |
| judge | WSL `judge_a800.sh` (Qwen3-VL on A800) |
| inversion | hupanwen `prepare_sharded.sh` (SAM2 masks) + `gravtrace` (replaces the old evaluator) |
| table | `vp_port/summarize_public211.py`, `make_table.py` |
