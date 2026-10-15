# Test scope and evidence

[简体中文](TESTING.zh-CN.md) · [README](README.md) · [Evidence index](validation/README.md)

## Evidence dates and release scope

Historical tests were performed on **2026-10-15**. Inference code and the existing regression tests in 0.1 are byte-identical to tested 0.1.1 / packaged 0.1.2; hashes are in [inference_baseline.json](validation/inference_baseline.json). Preparing this prerelease does not rerun checkpoint generation or imply a new GPU acceptance pass.

Historical regression logs remain labeled as historical. Any CPU-only checks performed for this package are recorded separately in `validation/prerelease_cpu_regressions.txt` and `validation/prerelease_cpu_environment.json`. A defined CI workflow is not evidence of a completed GitHub Actions run.

## Recorded environment

| Component | Full-checkpoint / actual-host tests |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16GB (16303 MiB usable reported) |
| System | Windows, 32GB RAM |
| ComfyUI | v0.40.0, `3541246a9c997f4b8c32f9f88f3c985d02021c79` |
| Python / torch | 3.11.9 / 2.9.1+cu130 |
| Diffusion weight | `flux1-krea-dev_fp8_scaled.safetensors` |
| Text encoders | `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors` |
| VAE | `ae.safetensors` |

The source-interface reference is ComfyUI `ba994879b88d0ebd9950b1e5c1e923d928f216ba`. Modified AReA core source is based on `9e38ec2d8457f2aa7800766e8a22d4960f160c69`. Historical independent regression environment: Python 3.12, torch 2.13.0+cpu, Pillow 11.3.0 and einops 0.8.1. Model bytes and SHA256 values are in [4k28_repeat.json](validation/4k28_repeat.json).

## Repeated 4K / 28-step ON/OFF measurements

4096×4096, batch=1, seed=0, AReA seed=0, Euler/simple, 28 steps, CFG=1, FluxGuidance=4.5, denoise=1, fixed shift=1.15. AReA: automatic 8+20 split, subsample, halo=8, bridge_ratio=0.5, window_batch_size=1. Tiled VAE: tile_size=512, overlap=64.

Prompt:

> A natural editorial photograph of an adult woman in a linen jacket standing beside a cafe window, soft daylight, realistic skin texture, relaxed expression, detailed fabric, coherent anatomy, a small glass bottle on the table, restrained colors, subtle film grain.

Each image used a fresh server process. Order: OFF→ON, then ON→OFF. Settings included PyTorch attention, `--reserve-vram 2`, `--disable-smart-memory`, `--cache-none`, and disabled previews, with unrelated custom nodes excluded. OS filesystem cache was not flushed. Startup/loading policies are part of the comparison and can affect results. The measurement observer was a test-only wrapper, not a production node in this package.

| Completed run | Sampling s | Workflow s | Sampling whole-card peak GiB | Torch allocated peak GiB |
|---|---:|---:|---:|---:|
| Pair 1 OFF | 958.83 | 981.43 | 14.49 | 5.78 |
| Pair 1 ON | 528.12 | 551.23 | 13.85 | 6.75 |
| Pair 2 ON | 529.21 | 552.19 | 15.20 | 6.75 |
| Pair 2 OFF | 957.65 | 981.53 | 14.56 | 5.78 |
| OFF median | **958.24** | **981.48** | **14.52** | **5.78** |
| ON median | **528.66** | **551.71** | **14.53** | **6.75** |

Pair speedups were 1.816× and 1.810×; median paired speedup was **1.813×**. OFF→ON sampling-time reduction based on medians was about 44.83%. Same-mode repeated runs had identical latent and final-pixel hashes. Two repeats measure repeatability for this configuration and do not increase prompt/seed coverage.

In pair 1, ON's sampling whole-card peak was about 651 MiB lower; in pair 2, it was about 659 MiB higher. Thus stable VRAM savings were **not demonstrated**. ON's highest whole-workflow observed card usage was 15.20 GiB. Torch allocated peak rose about 988 MiB in both pairs.

NVML samples were taken about every 0.1 s and include desktop/driver/other card usage; short peaks may be missed. Torch allocated peaks cover the sampler's visible allocations and exclude custom dynamic-VRAM allocations. The two metrics have different coverage and must not be added together or described as node-exclusive memory.

The observer's 1024×1024 / 2-step preflight produced exactly the same latent as standard KSampler. See [per-run evidence](validation/4k28_repeat.json), [CSV extract](validation/4k28_runs.csv) and [initial pair](validation/4k28_initial.json). The initial pair measured 963.78 / 524.88 s (1.836×); it used an earlier protocol and is kept separate from repeat statistics.

## Completed versus planned

| Group | Planned images | Completed | Status |
|---|---:|---:|---|
| P0 repeated cafe ON/OFF | 6 | 4 | Third OFF run interrupted at user request; partial run excluded |
| P1 first batch, hands/counts | 12 | 0 | Not started |

The suite was stopped because of its runtime. Cancellation is not counted as an algorithm success or failure. No continuation is scheduled by this package.

## Quality observations

Full images and crops were visually inspected by the development assistant; no independent human rating study was conducted. ON produced a single main portrait and a clearer bottle; OFF contained similar people/reflection-like structure and large foreground blur. Reflections alone were not classified as duplication errors. ON still had unnatural hands and fine white specks around glass, hands and some clothing. No obvious regular window seams were observed in that sample.

Apparent identity, age and pose changed. Same seed is not a same-person fidelity guarantee after changing attention. No multi-reviewer blind evaluation, FID/CLIP score, broad quality pass or comparison to generation-plus-upscaling was completed.

## Other historical checks

| Check | Result / scope |
|---|---|
| 27 CPU/reference regressions | Passed historically; includes true FLUX layer code using small random models |
| Actual ModelPatcher CPU and CUDA fixtures | Both attention phases, finite outputs, batch equivalence, branch isolation passed |
| 1280×1280 / 2 steps, full weights | Saved PNG; 24.58 s including load/polling; smoke only |
| 2048×2048 / 28 steps, full weights | Saved PNG; server 248.19 s, client 250.98 s; one sample inspected |
| 4096×4096 / 2 steps, full weights | Saved PNG; server 115.66 s, client 116.97 s; smoke only |
| Browser node loading and 2K opening | Historical installed 0.1.2 opened with no missing nodes and no queued prompt |
| 2K annotation in browser | Added five Note nodes and five groups; graph-to-prompt output and node titles unchanged; queue/history empty |
| 4K annotations | File-level checks for ON/OFF copies; titles, numeric parameters and links preserved; no images queued |
| Public portable workflows | Six UI/API pairs checked statically; no new live-browser import or generation claimed |

The 2K image had coherent main structure without obvious seams or saturated dots, but multiple bottles were present, so exact object-count accuracy was not established. Smoke tests validate execution, not normal-step image quality. Historical GPU/PNG/frontend records are indexed in [validation/README.md](validation/README.md).

Outstanding: P0's third pair, all P1 prompts, additional seeds, aspect ratios, full-checkpoint batch independence, LoRA, alternate quantization, compile, multi-GPU and editing/control combinations.

## Reproducing CPU checks

Use a separate development environment with torch, numpy, Pillow and einops. No model weights are required.

```bash
python tools/verify_release.py
python -m unittest discover -s tests -v
```

Without `AREA_COMFY_SOURCE`, 10 host-reference tests are explicitly skipped; the other 17 run. To run all 27, obtain ComfyUI source pinned to `ba994879b88d0ebd9950b1e5c1e923d928f216ba` and set the variable to its checkout. PowerShell example, using a sibling checkout:

```powershell
$env:AREA_COMFY_SOURCE = (Resolve-Path ../ComfyUI-reference).Path
python -m unittest discover -s tests -v
```

`tests/reference_host.py` stubs GPU management and optional kernel selection, then executes actual FLUX model/block/RoPE/wrapper code and selected attention functions from source. These are small CPU random-model tests, not a full ComfyUI server startup or trained-checkpoint generation. `AREA_TEST_TMP` can select an existing writable directory for temporary test files.

The GitHub Actions job installs CPU torch 2.9.1, checks out the pinned ComfyUI reference, runs static validation and all 27 CPU tests, and uploads a test log. Current remote results are recorded in [Actions](https://github.com/VividMuse-AGI/ComfyUI-AReA/actions), separately from the bundled historical local evidence.

## Evidence normalization

Historical local paths and test-server endpoints are replaced with portable labels; path separators in file/hash keys are normalized and model selections use basenames. The unrelated desktop-process list in the recorded nvidia-smi snapshot is omitted; its hardware/memory summary remains. Recorded metrics, hashes and run identities are retained. Historical labels such as `manual_review` or “single reviewer” describe the assistant's development-time visual checks, not an independent human assessment. Source and normalized record hashes are in [historical_sources.json](validation/historical_sources.json). The complete local raw archive, benchmark-only execution scripts and large image files are not included. The CSV is derived from the four successful repeated runs, without canceled samples.
