# physedit: generating and evaluating models with one command

`physedit` runs any set of registered video models on a PhysEdit benchmark, from manifests to the results table:

```bash
python3 -m physedit models                                              # the registered models and where they run
python3 -m physedit status public211 --models all                       # progress, ETAs, evaluation stages
python3 -m physedit run public211 --models ltx_i2v,cosmos3_nano         # one pass: start whatever can start
python3 -m physedit run public211 --models all --watch 10               # repeat every 10 min until all is done
python3 -m physedit run public211 --models ltx_i2v --limit 4            # trial: 4 rows per run, own namespace
python3 -m physedit run public211 --models all --dry-run                # print the plan, change nothing
```

Run it on the hub, the WSL environment of the workstation: it is the only machine that reaches every host
(hupanwen and wangzijun directly, the NSCC Starlight pods through the proxy in `~/.ssh/a800_fleet.conf`). It needs
only the Python 3 standard library there. For a long run, detach it:
`setsid nohup python3 -m physedit run public211 --models all --watch 10 > ~/.physedit/public211.log 2>&1 &`.

## What a pass does

Each pass reads the state of every host, then acts on it; nothing is kept on the hub except a lock file, so a pass
can be repeated at any time, and a run interrupted anywhere (a reboot, a recreated pod, a dropped connection)
continues from what is on disk.

1. **Survey.** Every host reports its GPUs' memory use and the physedit jobs it runs (each job's process carries its
   name); every site reports its run directories: manifest size, finished videos, worker start and end events.
2. **Prepare.** A model run that has not started is placed on the candidate site with the most free GPUs. The
   runners are deployed to `<site root>/tools`, the benchmark inputs (condition frames, prompts) are copied there once,
   and `build.py` writes the model's manifests from the benchmark's template rows: same sources, condition frames,
   prompts and seeds for every model; the model sets frame count, frame rate, canvas and output paths.
3. **Generate.** Every unfinished run keeps up to two workers on free GPUs: one walks the manifest forwards, one
   backwards, and every runner skips rows whose video exists, so they meet in the middle without coordination
   (DynamiCrafter's runner takes its whole manifest as one batch, so its workers get disjoint halves: `split`). A GPU
   is free when it holds less than the host's `busy_mib` and runs none of our jobs. A worker that dies is restarted
   on the next pass; one started three times within three hours is left alone and reported (`--retry` overrides).
4. **Copy home.** Videos made on a site the home host cannot read (NSCC) are copied to it as parallel tar streams
   through the hub. wangzijun's root is on the NFS share that hupanwen mounts, so its videos are read in place.
5. **Evaluate.** A finished run is published where the benchmark's evaluation scripts read it, and every stage whose
   input is ready starts: for public-211, V/P (segmentation, 1-s metrics, judge clips) after x5, the Qwen judge
   after V/P, the inversion masks after x1, then GravTrace. Stages marked exclusive run for one model at a time.
   When every stage of every model is done, the table is rebuilt.

`status` prints the same survey: per model and run the site, finished rows, seconds per row over the last 20
videos, ETA and live workers; per model the evaluation stages; and notes (waiting for GPUs, errors, restarts).

## Layout on every site

```
<root>/tools/                                  runners/*.py and build.py, deployed by the driver
<root>/<bench>/input/                          benchmark inputs copied from the home host (not on the home host)
<root>/<bench>[-<tag>]/<run>/<model>/
    spec.json  manifest.jsonl  manifest_w0.jsonl  manifest_w1.jsonl   (+ per-orientation manifests, prompts)
    generated/<row>/resized.mp4                       the video at the source resolution
    logs/w0.log  logs/w0.events                       runner output; start/end lines with time and exit code
<home root>/<bench>/eval/<model>/<stage>.log|.events  evaluation jobs
```

`--tag` (implied by `--limit`) gives a trial its own namespace; trials are generation only. `spec.json` records what
the manifests were built from: when a model's settings change, a run without videos is rebuilt and a run with videos
is reported (remove its directory to start it again), so settings are never mixed within a run.

## Configuration: `physedit/config.py`

- `SITES`: a filesystem (root, environment variables, and `vars` used in commands: interpreters, weights, code).
- `HOSTS`: ssh target, site, the GPUs physedit may use and the memory below which a GPU counts as idle. hupanwen GPUs
  1, 2, 6 and 7 are left to other users; NSCC `a800-4`/`a800-5` to the Qwen judge.
- `MODELS`: display name, frames, fps, canvas per orientation, template mode (i2v or v2v), GPUs per job, `split`,
  extra manifest fields, and per site the command that generates one worker's manifest (`{manifest}`, `{dir}`,
  `{gpu}`, `{tools}`, `{run}`, `{worker}` plus the site's and the host's `vars`; the NSCC pods run two container
  images, so their interpreter is a host variable).
- `BENCHMARKS`: home site, runs, template manifests per mode and run, inputs to copy, where finished runs are
  published, the evaluation stages (command, prerequisite, GPUs, exclusive, hub or home) and the table command.

To add a model, add an entry to `MODELS` (and its runner to `physedit/runners/` if no existing one fits); to add a
machine, a `HOSTS` entry (and a `SITES` entry for a new filesystem).

## Runners (`physedit/runners/`)

The generation scripts used for the September runs, collected here and deployed by the driver: `diffusers_i2v.py`
(LTX, CogVideoX, Wan 2.2 5B, HunyuanVideo 1.5), `cosmos_predict2.py`, `dynamicrafter.py` + `crop_letterbox.py`,
`videogpa_wan22.py` + `videogpa_finalize.py`, `physrvg.py`, `cosmos3.py`, `wan22_a14b.py` (Wan 2.2 A14B and PhysAlign),
`minimax_h3.py`. Each reads a manifest row's condition image, prompt and seed and writes `resized.mp4` at the source
size. Changes from the September copies: `wan22_a14b.py` records a failed row and continues (a single corrupt write
had stopped a run), `minimax_h3.py` takes the model path as an argument and writes where the manifest says.

## Status and limitations

- public-211 evaluation calls the UniversalPhysicsEval port on hupanwen (`/data2/zhangshuran/tmp/public211_20260920`),
  whose paths are fixed to the public-211 run directories. Models evaluated before physedit are recognised and left
  untouched; a new model is evaluated next to them, but the table script lists only the September 12 models.
- The PhysEditing dataset (`phyeditingvideo/Phyediting` on the Hub) will become a benchmark entry: its template rows,
  inputs and evaluation stages, in the dataset's own format.
- MiniMax-H3 needs a host with two GPUs; none is configured yet.
