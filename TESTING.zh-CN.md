# 验证范围与实测证据

[English](TESTING.md) · [使用说明](README.zh-CN.md) · [证据索引](validation/README.md)

## 记录日期与版本范围

历史测试日期为 **2026-10-15**。本预发布的推理代码和既有测试与实测 0.1.1、已打包 0.1.2 逐字节一致，见 [代码哈希](validation/inference_baseline.json)。整理本版没有重跑完整权重或 GPU 出图。

历史回归日志保留历史标记；本次本地 CPU 检查单独记录在 `validation/prerelease_cpu_regressions.txt` 与 `validation/prerelease_cpu_environment.json`。配置了 CI 不代表已经在 GitHub 上运行通过。

## 实机环境

RTX 5080 16GB（可用容量记录为 16303 MiB）、Windows、32GB 内存；ComfyUI v0.40.0，提交 `3541246a9c997f4b8c32f9f88f3c985d02021c79`；Python 3.11.9，torch 2.9.1+cu130。权重为 `flux1-krea-dev_fp8_scaled.safetensors`，编码器为 `clip_l.safetensors`、`t5xxl_fp8_e4m3fn.safetensors`，VAE 为 `ae.safetensors`。权重字节数和 SHA256 见 [重复测试 JSON](validation/4k28_repeat.json)。

ComfyUI 开发接口基线为 `ba994879b88d0ebd9950b1e5c1e923d928f216ba`，上游 AReA 基线为 `9e38ec2d8457f2aa7800766e8a22d4960f160c69`。历史独立 CPU 回归使用 Python 3.12、torch 2.13.0+cpu、Pillow 11.3.0、einops 0.8.1。

## 4K / 28 步重复对照

固定 4096×4096、batch=1、KSampler seed=0、AReA seed=0、Euler/simple、28 步、CFG=1、FluxGuidance=4.5、denoise=1、shift=1.15。AReA 自动 8+20、subsample、halo=8、bridge_ratio=0.5、window_batch_size=1。VAE tile_size=512、overlap=64。

原始提示词：

> A natural editorial photograph of an adult woman in a linen jacket standing beside a cafe window, soft daylight, realistic skin texture, relaxed expression, detailed fabric, coherent anatomy, a small glass bottle on the table, restrained colors, subtle film grain.

每张图片用新服务进程，顺序为 OFF→ON，然后 ON→OFF。使用 PyTorch attention、`--reserve-vram 2`、`--disable-smart-memory`、`--cache-none`，关闭预览并排除无关自定义节点；没有清除系统文件缓存。加载和 offload 策略会影响结果。计时观察器是测试专用包装器，未随本包作为生产节点安装。

| 两次中位数 | OFF | ON |
|---|---:|---:|
| 采样循环 | 958.24 秒 | 528.66 秒 |
| 全工作流，不含服务启动 | 981.48 秒 | 551.71 秒 |
| 采样区间整卡峰值 | 14.52 GiB | 14.53 GiB |
| PyTorch allocated 峰值 | 5.78 GiB | 6.75 GiB |

两对加速比分别为 1.816、1.810，中位约 **1.813 倍**；按耗时中位数计算，采样时间减少约 44.83%。OFF 两次 958.83 / 957.65 秒，ON 两次 528.12 / 529.21 秒；同模式的 latent 和最终像素哈希一致。两个重复样本只说明该配置的重复性，没有增加题材或种子覆盖。

第一对 ON 的整卡采样峰值低约 651 MiB，第二对高约 659 MiB，故**未证明稳定节省显存**。ON 的单次整卡全流程最高观测值达 15.20 GiB；Torch allocated 峰值两次均增加约 988 MiB。

NVML 约每 0.1 秒采样，包含桌面、驱动等使用量，可能漏掉瞬时峰值。Torch 统计仅覆盖采样器可见分配，不含 ComfyUI 自定义动态分配。二者口径不同，不能相加或当作节点独占显存。

观察器在 1024×1024 / 2 步预检中与标准 KSampler 的 latent 逐值一致。逐次数据见 [JSON](validation/4k28_repeat.json) 和 [CSV](validation/4k28_runs.csv)。更早的一对测试为 963.78 / 524.88 秒、1.836 倍，采用较早协议，见 [初次记录](validation/4k28_initial.json)，未并入重复测试统计。

## 计划与实际完成范围

| 测试组 | 计划 | 完成 | 状态 |
|---|---:|---:|---|
| P0 窗边人像重复对照 | 6 张 | 4 张 | 第三次 OFF 中途按用户要求停止，半途样本排除 |
| P1 首批，手部与计数 | 12 张 | 0 张 | 未开始 |

因运行耗时停止余下测试，取消样本不计为算法成功或失败，本包不会调度继续执行。

## 画质观察

开发助手曾查看完整图和局部，没有组织独立人工评分：ON 形成单主体窗边构图，瓶子较清楚；OFF 有相似人物或反射感，以及大面积前景模糊遮挡。仅凭反射未判定人物复制错误。ON 的手部仍不自然，玻璃、手部及部分衣物上有细碎白点，该样例未见明显规则窗口接缝。

人物身份感、年龄感和姿态随注意力变化；相同 seed 不等于同一人物保真对照。未完成多评审盲评、FID/CLIP 分数、多题材画质通过结论，或低分辨率生成后放大方案的比较。

## 其他历史验收

- 27 项历史 CPU/参考层回归通过：调度、池化、独立 batch、窗口 attention、桥接、并发防覆盖与元数据等。
- 实际 ModelPatcher 的 CPU/CUDA 小型随机模型通过两阶段、有限输出、batch 一致性及模型分支隔离检查。
- 完整 FP8：1280×1280 / 2 步 24.58 秒，2048×2048 / 28 步客户端 250.98 秒、服务端 248.19 秒，4096×4096 / 2 步客户端 116.97 秒、服务端 115.66 秒，均成功保存 PNG。2 步仅验执行，不作画质通过依据。
- 2K 单张主体结构完整，未见明显接缝或彩色点状伪影，但存在多个瓶子，未据此验证精确计数。
- 后续已验证安装版 0.1.2 浏览器加载、打开 2K 无缺失节点；添加五个便签和五个分组前后执行 prompt 与节点标题一致，队列及历史为空。
- 4K 开关版完成文件级注释核对，标题、数值参数及连接保持一致，没有排队出图。
- 本版六份通用 UI/API 工作流仅作静态核对，未重新导入真实浏览器或出图。

证据见 [validation 索引](validation/README.md)。未完成项目包括 P0 第三对、P1 全部、多种子、长宽比、完整权重 batch、LoRA、其他量化、compile、多卡及编辑控制组合。

## CPU 检查复现

在独立开发环境安装 torch、numpy、Pillow、einops，无需模型权重：

```bash
python tools/verify_release.py
python -m unittest discover -s tests -v
```

不设 `AREA_COMFY_SOURCE` 时执行 17 项，另外 10 项宿主源码测试明确跳过。全部 27 项需要取得固定提交 `ba994879b88d0ebd9950b1e5c1e923d928f216ba` 的 ComfyUI 源码，并将环境变量指向该目录。PowerShell 同级源码示例：

```powershell
$env:AREA_COMFY_SOURCE = (Resolve-Path ../ComfyUI-reference).Path
python -m unittest discover -s tests -v
```

`tests/reference_host.py` 替换 GPU 管理和可选内核选择等宿主服务，执行真实 FLUX 模型、块、RoPE、wrapper 以及选取的 attention 源码。它是 CPU 小型随机模型测试，不是完整 ComfyUI 启动或训练权重验收。可用 `AREA_TEST_TMP` 指定已有的可写临时目录。

CI 定义使用 CPU torch 2.9.1，检出固定 ComfyUI 源码，执行静态检查与 27 项 CPU 回归并上传日志。当前远程结果见 [Actions](https://github.com/VividMuse-AGI/ComfyUI-AReA/actions)，与包内历史本地测试记录分开保存。

## 公开记录整理方式

历史绝对路径和本机服务地址替换为通用标记，文件/哈希键的分隔符统一为 `/`，模型选择统一为文件名；nvidia-smi 历史快照删除了无关桌面进程列表，保留硬件及显存摘要。测量值、哈希、运行编号保留。历史字段中的 `manual_review` 或“单人复核”指开发助手查看图像，不代表独立人工评审。原记录与整理后记录的 SHA256 见 [historical_sources.json](validation/historical_sources.json)。完整本地原始归档、测试专用执行脚本和大图未包含；CSV 只提取四次成功运行，排除取消样本。
