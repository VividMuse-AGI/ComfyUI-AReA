# v0.1 — experimental prerelease

**Experimental prerelease.** Public version is `0.1`, with tag `v0.1`. Repository: [VividMuse-AGI/ComfyUI-AReA](https://github.com/VividMuse-AGI/ComfyUI-AReA). No Comfy Registry entry is provided.

Independent ComfyUI port of AReA for native FLUX.1 / FLUX.1-Krea-dev text-to-image sampling. The acceptance checkpoint is Krea FP8 scaled. This package includes the MODEL patch, collision-safe PNG saving, 2K/4K annotated workflows in English and Chinese, existing test evidence and CPU development checks.

Recorded RTX 5080 16GB results, 4096×4096 / 28 steps, two repeated pairs: median sampling **958.24 s OFF → 528.66 s ON**, paired speedup **about 1.81×**. Whole-card sampling peak medians were **14.52 / 14.53 GiB**. Stable VRAM savings and general quality gains are not established; coverage is one prompt/seed, with remaining hand and white-speck artifacts. P0 completed 4/6 images; P1 completed 0/12.

Inference code is unchanged from the tested local baseline. Public workflows use model basenames: reselect installed diffusion, CLIP and VAE files after import. Default node names are preserved. Package preparation runs static/CPU checks only; no new checkpoint generation or remote CI result is claimed.

Source: Apache-2.0, with upstream attribution in NOTICE. Model weights are not included and retain their own licenses. Read README.md and TESTING.md before experimenting. Download `ComfyUI-AReA-v0.1.zip` and its `.sha256` file from the release assets; extract the `ComfyUI-AReA` folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

---

# v0.1 — 实验性预发布

**实验性预发布。** 对外版本为 `0.1`，标签为 `v0.1`。仓库：[VividMuse-AGI/ComfyUI-AReA](https://github.com/VividMuse-AGI/ComfyUI-AReA)。未提供 Comfy Registry 条目。

面向原生 FLUX.1 / FLUX.1-Krea-dev 文生图的独立 AReA 移植，完整权重验收使用 Krea FP8 scaled。包含模型补丁、防覆盖保存、中英文 2K/4K 注释工作流、已有测试证据和 CPU 开发检查。

RTX 5080 16GB、4096×4096、28 步、两对重复实测：采样中位耗时从 **958.24 秒降至 528.66 秒**，配对加速约 **1.81 倍**；整卡采样峰值中位数为 **14.52 / 14.53 GiB**。尚未证实稳定省显存或普遍提升画质，画质只覆盖一个提示词与种子，仍有手部和细碎白点。P0 完成 4/6 张，P1 完成 0/12 张。

推理代码沿用实测基线；公开工作流使用通用文件名，导入后需重新选择模型、CLIP、VAE。节点名称保持默认。本次仅做静态及 CPU 检查，没有新出图，也没有远程 CI 运行结果。

代码采用 Apache-2.0，保留原作者归属；不包含权重。下载 Release 附件中的 `ComfyUI-AReA-v0.1.zip` 和 `.sha256` 校验文件，将包内 `ComfyUI-AReA` 文件夹解压到 `ComfyUI/custom_nodes/`，然后重启 ComfyUI。
