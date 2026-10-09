"""Fleet, models and benchmarks. Adding a model, a machine or a benchmark only touches this file.

A model's `cmd` maps a site to the shell command that generates the rows of one manifest there. Placeholders are
filled from the site's `vars` and, per job, with
    {manifest}  the worker's manifest, {dir}/manifest_{worker}.jsonl (w0 forwards, w1 backwards or the other half)
    {worker}    w0 or w1, for commands that read other per-worker files ({dir}/prompts_{worker}.json)
    {dir}       the model's run directory on that site (manifests, generated/<row>/resized.mp4, logs/)
    {tools}     the deployed physedit runners on that site
    {gpu}       the job's GPU index(es), comma separated (also exported as CUDA_VISIBLE_DEVICES)
    {run} {job}
Every runner skips rows whose resized.mp4 already exists, so any job can be stopped and started again.
"""
import os

SSH = "ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=30 -o ServerAliveCountMax=4"
NSCC_SSH = SSH + " -F " + os.path.expanduser("~/.ssh/a800_fleet.conf")   # Starlight pods behind the NSCC proxy
ENV = "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1"

# A site is one filesystem: all its hosts see the same paths. `shared_with`: the evaluation host reads this site's
# files at the same paths (NFS), so its videos need no copy.
Z = "/XYAIFS00/HOME/sysu_xdliang/sysu_xdliang_3/HDD_POOL/zhuangshuran"
V3 = "/21231_data1/zhangshuran/video_models_3090"
PS = "/21231_data1/zhangshuran/physics_specialized_20260811"
SITES = {
    "hupanwen": dict(root="/data2/zhangshuran/physedit", env=ENV + " PYTHONNOUSERSITE=1", vars=dict(
        python="/data1/zhangshuran/miniconda3/envs/cosmos-predict2/bin/python",
        py_physinv="/data1/zhangshuran/miniconda3/envs/physinv/bin/python",
        py_wan="/data2/zhangshuran/envs/ltx_v2v/bin/python",
        py_hunyuan="/data2/zhangshuran/public211_hunyuan_20260926/venv/bin/python",
        models="/data2/zhangshuran/models",
        hunyuan="/data2/zhangshuran/models/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        cosmos_tools="/data2/zhangshuran/cosmos_predict2_tools",
        cosmos_model="/data1/zhangshuran/models/Cosmos-Predict2-2B-Video2World")),
    "wangzijun": dict(root="/21231_data1/zhangshuran/physedit", shared_with="hupanwen",
                      env=ENV + " PYTHONNOUSERSITE=1 HF_HUB_CACHE=/home/wangzijun_p25/.cache/huggingface/hub", vars=dict(
        python="/data1/zhangshuran/miniconda3/envs/cosmos-predict2/bin/python",
        py_hunyuan=f"{V3}/envs/hunyuan15/bin/python",
        hunyuan=f"{V3}/weights/hunyuanvideo15_480p_i2v_step_distilled",
        py_dc=f"{V3}/envs/dynamicrafter_py38_libmamba/bin/python",
        dc_repo=f"{V3}/repos/DynamiCrafter",
        dc_ckpt=f"{V3}/weights/dynamicrafter_1024/model.ckpt",
        py_lora="/data1/zhangshuran/envs/lora_tf514/bin/python",
        physics=PS,
        wan5b="/data1/zhangshuran/models/Wan2.2-TI2V-5B",
        wan5b_diffusers="/data1/zhangshuran/models/Wan2.2-TI2V-5B-Diffusers")),
    # the container's python3 needs the account's user site-packages, so no PYTHONNOUSERSITE here
    "nscc": dict(root="/XYAIFS00/HOME/sysu_xdliang/sysu_xdliang_3/HDD_POOL/zhangshuran/physedit", env=ENV, vars=dict(
        python="python3",                                    # the driver's helpers need only the standard library
        wan_code=f"{Z}/wan22_a14b_a800_smoke_20260831/code/Wan2.2-official",
        wan_a14b=f"{Z}/Wan2.2-I2V-A14B-official-bf16",
        physalign=f"{Z}/PhysAlign-Wan2.2-I2V-A14B-community/adapter",
        cosmos3=f"{Z}/Cosmos3-Nano",
        h3="/XYAIFS00/HDD_POOL/sysu_xdliang/sysu_xdliang_3/zhangshuran/models/MiniMax-H3-FL2VA-diffusers-20260903")),
}

# GPUs a job may take. A GPU is free when it holds at most `busy_mib` and runs none of our jobs (other users'
# small processes, e.g. Isaac Sim's ~0.9 GB on every wangzijun GPU, are tolerated). A host's `vars` override its
# site's: the NSCC pods share one disk but run two container images. On the pytorch-ng (NGC) image the container's
# python3 3.12 has torch 2.7 and the H3 bundle's diffusers 0.40-dev (Cosmos3OmniPipeline, ModularPipeline); the
# uv venv below links to /usr/bin/python3 and only works on the deeplearni image (Python 3.10), where Cosmos3 ran in
# September.
NGC = dict(py_models="python3")
DEEPLEARNI = dict(py_models=f"{Z}/envs/cosmos3_diffusers_py310/bin/python")
HOSTS = {
    "hupanwen": dict(site="hupanwen", ssh=SSH + " hupanwen_t_server", gpus=[0, 3, 4, 5], busy_mib=1500),  # 1/2/6/7: others
    "wangzijun": dict(site="wangzijun", ssh=SSH + " wangzijun_p25_server", gpus=range(6), busy_mib=3000),
    # the deeplearni pods a800-4 and a800-5 (vars=DEEPLEARNI) are left to the Qwen judge (~/judge_a800.sh)
    **{n: dict(site="nscc", ssh=NSCC_SSH + " " + n, gpus=[0], busy_mib=1500, vars=NGC)
       for n in ("a800-1", "a800-2", "a800-3", "h100-1", "a100-1")},
}

CANVAS_848 = {"landscape": (848, 480), "portrait": (480, 848), "square": (640, 640)}
CANVAS_832 = {"landscape": (832, 480), "portrait": (480, 832), "square": (640, 640)}


def diffusers(python, weights, pipeline, args):
    """the shared diffusers image-to-video runner (LTX, CogVideoX, Wan 5B, HunyuanVideo)"""
    return (python + " {tools}/diffusers_i2v.py --generation-manifest {manifest} --model-dir " + weights
            + " --pipeline " + pipeline + " --cuda-visible-devices {gpu} --torch-dtype bfloat16 --skip-existing " + args)


A14B = ("cd {wan_code} && python3 {tools}/wan22_a14b.py --code-dir {wan_code} --manifest {manifest} --ckpt-dir {wan_a14b}"
        " --status-jsonl {dir}/logs/{worker}_status.jsonl --state-json {dir}/logs/{worker}_state.json")

# name: label in the result tables; frames/fps: what the model generates; canvas: generation size per source
# orientation (else the runner's own); mode: which benchmark template the rows come from (i2v or v2v); gpus: per job;
# split: the runner takes its whole manifest as one batch, so the two workers get disjoint halves;
# fields: extra manifest fields, optionally per run (strings are formatted with the row and {inputs[name]});
# sec_per_row: expected generation time, for ETAs before the first rows finish.
MODELS = {
    # LTX and CogVideoX need sides divisible by 32: aspect-preserving canvases as in September
    "ltx_i2v": dict(name="LTX-Video-2B", frames=81, fps=16, sec_per_row=46,
        canvas={"landscape": (704, 480), "portrait": (480, 704), "square": (480, 480)}, cmd={"hupanwen": diffusers(
        "{python}", "{models}/LTX-Video-diffusers", "ltx_i2v", "--num-inference-steps 30 --guidance-scale 3.0 --num-frames 81")}),
    "cogvideox_i2v": dict(name="CogVideoX1.5-5B-I2V", frames=81, fps=16, sec_per_row=514,
        canvas={"landscape": (768, 480), "portrait": (480, 768), "square": (480, 480)}, cmd={"hupanwen": diffusers(
        "{python}", "{models}/CogVideoX1.5-5B-I2V", "cogvideox_i2v", "--num-inference-steps 50 --guidance-scale 6.0 --num-frames 81")}),
    "wan_ti2v5b": dict(name="Wan2.2-TI2V-5B", frames=81, fps=16, sec_per_row=210, cmd={"hupanwen": diffusers(
        "{py_wan}", "{models}/Wan2.2-TI2V-5B-Diffusers", "wan_i2v", "--num-inference-steps 40 --guidance-scale 3.5"
        " --height 480 --width 832 --num-frames 81 --wan-invert-output --offload-mode model")}),
    "cosmos": dict(name="Cosmos-Predict2-2B", frames=81, fps=16, mode="v2v", sec_per_row=450, cmd={"hupanwen":
        "{python} {tools}/cosmos_predict2.py --generation-manifest {manifest} --cosmos-tools {cosmos_tools}"
        " --python {python} --model-dir {cosmos_model} --cuda-visible-devices {gpu} --skip-existing"}),
    "hunyuan15_i2v": dict(name="HunyuanVideo-1.5-I2V", frames=121, fps=24, canvas=CANVAS_848, sec_per_row=900,
        cmd=dict.fromkeys(("hupanwen", "wangzijun"), diffusers("{py_hunyuan}", "{hunyuan}", "hunyuan15_i2v",
            "--num-frames 121 --num-inference-steps 12 --guidance-scale 1.0 --offload-mode sequential"))),
    "dynamicrafter_i2v": dict(name="DynamiCrafter-1024", frames=16, fps=8, letterbox=True, split=True, sec_per_row=60,
        cmd={"wangzijun":
        "{py_dc} {tools}/dynamicrafter.py --generation-manifest {manifest} --repo-dir {dc_repo} --checkpoint {dc_ckpt}"
        " --cuda-visible-devices {gpu} --ddim-steps 50 --skip-existing && {python} {tools}/crop_letterbox.py {manifest}"}),
    "videogpa_ti2v5b": dict(name="VideoGPA-Wan2.2-TI2V-5B", frames=49, fps=15, prompts_json=True, sec_per_row=150, cmd={"wangzijun":
        "PYTHONPATH={physics}/repos/VideoGPA/Wan2.2:{physics}/deps/videogpa {py_lora} {tools}/videogpa_wan22.py"
        " --model_path {wan5b} --prompt_json {dir}/prompts_{worker}.json --output_dir {dir}/raw"
        " --lora_path {physics}/weights/VideoGPA-Wan2.2TI2V-lora --lora_weight 0.2 --gpu_id 0 --seed 20260920"
        " --frame_num 49 --max_area 399360 --sampling_steps 20 --fps 15 --offload_model"
        " && {python} {tools}/videogpa_finalize.py {manifest} {dir}/raw 20260920"}),
    # PhysRVG takes one canvas per process; x5 conditions on the 5 source frames ending at the condition frame
    "physrvg_ti2v5b": dict(name="PhysRVG-Wan2.2-TI2V-5B", frames=49, fps=15, canvas=CANVAS_832, split_orientation=True,
        sec_per_row=150, fields={"x5": {"source_video": "{inputs[prefix5]}/{source_sample_id}.mp4"}}, cmd={"wangzijun":
        "for o in landscape portrait square; do [ -s {dir}/manifest_{worker}_$o.jsonl ] || continue;"
        " case $o in landscape) H=480 W=832;; portrait) H=832 W=480;; square) H=640 W=640;; esac;"
        " PYTHONPATH={physics}/repos/PhysRVG:{physics}/deps/physrvg {py_lora} {tools}/physrvg.py"
        " --manifest {dir}/manifest_{worker}_$o.jsonl --model-id {wan5b_diffusers}"
        " --lora-checkpoint {physics}/weights/PhysRVG/lora/checkpoint --model-label PhysRVG-Wan2.2-TI2V-5B"
        " --output-root {dir} --status-jsonl {dir}/logs/{worker}_status_$o.jsonl --device 0 --height $H --width $W"
        " --num-frames 49 --num-inference-steps 16 --fps 15 --offload-mode model"
        " --condition-policy $([ {run} = x5 ] && echo source_history5 || echo static_frame) || exit 1; done"}),
    "cosmos3_nano": dict(name="Cosmos3-Nano", frames=81, fps=16, canvas=CANVAS_832, sec_per_row=480,
        fields={"cosmos_num_inference_steps": 35, "cosmos_guidance_scale": 6.0}, cmd={"nscc":
        "{py_models} {tools}/cosmos3.py --model {cosmos3} --manifest {manifest} --node {job}"
        " --status-jsonl {dir}/logs/{worker}_status.jsonl --state-json {dir}/logs/{worker}_state.json"
        " --ffmpeg $({py_models} -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"}),
    "wan22_i2v_a14b": dict(name="Wan2.2-I2V-A14B", frames=81, fps=16, sec_per_row=870, fields={"sample_steps": 40},
        cmd={"nscc": A14B}),
    "physalign_wan22_i2v_a14b": dict(name="PhysAlign-Wan2.2-I2V-A14B", frames=81, fps=16, sec_per_row=870,
        fields={"sample_steps": 40}, cmd={"nscc": A14B + " --physalign-adapter-dir {physalign}"}),
    "minimax_h3": dict(name="MiniMax-H3", frames=124, fps=24, gpus=2, sec_per_row=1200,
        canvas={"landscape": (896, 512), "portrait": (512, 896), "square": (640, 640)}, cmd={"nscc":
        "{py_models} {tools}/minimax_h3.py --source-manifest {manifest} --out-root {dir}/logs/{worker} --model {h3} --steps 50"}),
}

# A benchmark: the site holding its inputs (the evaluation host), the template rows every model's manifest is
# derived from (per mode and run: same sources, condition frames, prompts and seeds), the input directories other
# sites need, where finished runs are published for the evaluation scripts, and the evaluation stages.
RUN = "/data2/zhangshuran/UniversalPhysicsEval_runs/public211_full2prompt_20260920"
INV = "/data2/zhangshuran/UniversalPhysicsEval_runs/public211_inv_x1_20260922"
P211 = "/data2/zhangshuran/tmp/public211_20260920"          # public-211 evaluation scripts (UniversalPhysicsEval port)
GT2 = "/data2/zhangshuran/GravTrace2"                         # gravtrace working copy with the ground-truth scenes
BENCHMARKS = {
    "public211": dict(
        home="hupanwen",
        runs={"x5": "V/P run, condition frame aligned with self100", "x1": "inversion run, first visible frame"},
        templates={"i2v": {"x5": f"{RUN}/ltx_i2v/manifests/ltx_i2v_generation_manifest.jsonl",
                           "x1": f"{INV}/ltx_i2v/manifests/ltx_i2v_generation_manifest.jsonl"},
                   "v2v": {"x5": f"{RUN}/cosmos/manifests/cosmos_generation_manifest.jsonl",
                           "x1": f"{INV}/cosmos/manifests/cosmos_generation_manifest.jsonl"}},
        inputs={"x5_conditioning": f"{RUN}/conditioning", "x1_conditioning": f"{INV}/conditioning",
                "prompts": f"{RUN}/prompts", "prefix5": f"{P211}/prefix5"},
        publish={"x5": RUN + "/{model}/manifests/{model}_generation_manifest.jsonl",
                 "x1": INV + "/{model}/manifests/{model}_generation_manifest.jsonl"},
        # models evaluated before physedit; the V/P segmentation step rebuilds its job list over all models
        legacy=["ltx_i2v", "cosmos", "cogvideox_i2v", "wan_ti2v5b", "cosmos3_nano", "physrvg_ti2v5b", "videogpa_ti2v5b",
                "dynamicrafter_i2v", "hunyuan15_i2v", "wan22_i2v_a14b", "physalign_wan22_i2v_a14b"],
        # stage: after (a run that must be published, or an earlier stage), where (home or hub), gpus,
        # exclusive (one model at a time: shared job lists). {vp_models}: every published x5 model.
        evaluate=[
            dict(stage="vp", after="x5", gpus=3, exclusive=True, cmd=(
                f"{{python}} {P211}/vp_port/build_jobs_public211.py {{vp_models}} && rm -f {RUN}/vp_metrics/supervisor.lock"
                f" && cd {P211}/vp_port && {{py_physinv}} run_repair_public211.py --gpus {{gpu}} --cpu-workers 6"
                f" && {{py_physinv}} score_window_1s_public211.py {{model}} > {RUN}/vp_metrics/metrics_1s_{{model}}.json"
                f" && {{python}} build_qwen_proxy_public211.py {{model}} > {RUN}/vp_metrics/qwen_v4_1s.{{model}}.json")),
            dict(stage="judge", after="vp", where="hub", cmd="~/judge_a800.sh {model}"),
            dict(stage="inversion", after="x1", gpus=3, exclusive=True,
                 cmd=f"{P211}/run_inv_eval_model_sharded.sh {{model}} 3 {{gpu}}"),
            dict(stage="gravtrace", after="inversion", cmd=(
                f"cd {GT2} && {{py_physinv}} runs/p211_videos.py {INV} {{model}} manifests/gt260.jsonl runs/p211/videos_{{model}}.jsonl"
                f" && {{py_physinv}} scripts/generated_manifest.py manifests/gt260.jsonl runs/p211/videos_{{model}}.jsonl runs/p211/manifest_{{model}}.jsonl"
                f" && {{py_physinv}} -m gravtrace fit runs/p211/manifest_{{model}}.jsonl -o runs/p211/pred_{{model}}.jsonl --workers 16"
                f" && {{py_physinv}} -m gravtrace score runs/p211/manifest_{{model}}.jsonl runs/p211/pred_{{model}}.jsonl -o runs/p211/summary_{{model}}.json")),
        ],
        table=(f"S={RUN}/vp_metrics/summary; cd {P211}/vp_port && {{python}} summarize_public211.py > $S/summarize.log"
               f" && {{python}} {P211}/inv_pscore.py $S/public211_inv_pscore.json > $S/public211_inv_pscore.txt"
               f" && {{python}} {P211}/make_table.py $S/public211_vp_table.json $S/public211_inv_pscore.json $S/public211_vp_inv_table"),
    ),
}
