# ComfyUI-AReA

[English](README.md) · [工作流](workflows/README.md) · [实测记录](TESTING.zh-CN.md) · [预发布说明](RELEASE_NOTES.md) · [最终审核](FINAL_AUDIT.zh-CN.md)

AReA 高分辨率注意力方法的独立 ComfyUI 实验性移植。通过 MODEL → MODEL 补丁节点，将粗、细两阶段注意力接入 ComfyUI 原生 FLUX.1 采样。

**当前对外版本：`0.1`，实验性预发布。** 推理代码与已实测的本机开发版 0.1.1、0.1.2 相同；后两者是内部编号，不是已发布版本。本版补齐公开文档、通用模型路径、带注释工作流及历史测试记录。[下载 v0.1](https://github.com/VividMuse-AGI/ComfyUI-AReA/releases/tag/v0.1)。未提供 Comfy Registry 条目。

## 已验证的价值

- FLUX.1-Krea-dev FP8 scaled 完整权重已完成 2048×2048 / 28 步出图，以及 4096×4096 / 28 步开关对照。
- RTX 5080 16GB 上，两对 4K 重复测试的采样中位耗时从 **958.24 秒降至 528.66 秒**，配对加速比中位约 **1.81 倍**；全工作流中位耗时为 981.48 / 551.71 秒。结果限定于记录中的环境与启动配置。
- **尚无稳定节省显存的证据。** 关闭 / 开启的整卡采样峰值中位数为 14.52 / 14.53 GiB，PyTorch allocated 峰值为 5.78 / 6.75 GiB。
- 画质只覆盖一个窗边人像提示词和一个种子。开启后仍有手部和细碎白点问题，人物身份感、姿态及构图也会改变，不能据此认定普遍提升画质。

![历史4K对照：左关闭，右开启](validation/4k_comparison.jpg)

图片是已有测试的缩小预览。原生 4K 图片和连续遥测保留于原本地测试归档，未随仓库提供；包内提供逐次指标与输出哈希。验证范围见 [TESTING.zh-CN.md](TESTING.zh-CN.md)。

## 安装

1. 将整个仓库文件夹解压到 `ComfyUI/custom_nodes/ComfyUI-AReA/`，确保 `__init__.py` 直接位于该目录。替换旧版前，先把旧文件夹移到 `custom_nodes` 之外，避免重复加载。
2. ComfyUI 需要 FLUX `post_input`、`attn1_patch`、`optimized_attention_override`、diffusion wrapper 和 `sample_sigmas` 接口。完整权重实测环境为 **ComfyUI v0.40.0**，提交 `3541246a9c997f4b8c32f9f88f3c985d02021c79`、Python 3.11.9、torch 2.9.1+cu130。开发接口基线见测试文档；未宣称所有历史或未来版本都兼容。
3. 正常 ComfyUI 环境已有 torch、numpy、Pillow。若缺包，只用实际启动 ComfyUI 的 Python 补装缺少的包，不要为此重新安装 CUDA/PyTorch。
4. 重启 ComfyUI，搜索 `AReA`。

节点不会联网下载模型或安装软件。包内不含权重，权重仍需遵守各自许可证。

## 模型范围

适配目标为原生 **FLUX.1-Krea-dev / FLUX.1-dev 标准文生图架构**。完整模型验收使用 Krea FP8 scaled；其他 checkpoint 和精度未逐一验收。当前适配范围不含 Krea 2、FLUX.2、Qwen-Image 或 Kontext 编辑。

| 加载器 | 工作流示例文件名 | 模型目录 |
|---|---|---|
| Load Diffusion Model / `UNETLoader` | `flux1-krea-dev_fp8_scaled.safetensors` | `models/diffusion_models/` |
| Load CLIP (Dual) / `DualCLIPLoader` | `clip_l.safetensors`、`t5xxl_fp8_e4m3fn.safetensors` | `models/text_encoders/` |
| Load VAE / `VAELoader` | `ae.safetensors` | `models/vae/` |

**导入工作流后，在以上三个加载器中重新选择自己已安装的模型文件。** 公开工作流只写文件名，避免绑定本机子目录；若权重放在子目录中，请选择 ComfyUI 实际列出的条目。原模型见 [FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)。Diffusers 分片文件不能直接作为单个 ComfyUI 权重加载。

## 带说明的工作流

| 中文注释工作流 | 画布 | AReA |
|---|---|---|
| [2K 开启](workflows/zh-CN/01_Krea_FP8_2K_AReA_ON.json) | 2048×2048 | true |
| [4K 开启](workflows/zh-CN/02_Krea_FP8_4K_AReA_ON.json) | 4096×4096 | true |
| [4K 关闭对照](workflows/zh-CN/03_Krea_FP8_4K_AReA_OFF.json) | 4096×4096 | false |

建议从 2K 开始。拖入不带 `_API` 的 JSON；同名 `_API.json` 用于程序提交。英文注释版位于 `workflows/`。六份 UI 工作流均保留默认节点名称，通过标准 Note 便签和彩色分组标记关键位置。插件自身提供的节点显示文字目前在两种工作流中均为中文。

| 要修改的内容 | 位置 |
|---|---|
| 提示词 | 节点 5：有内容的 `CLIPTextEncode`；第二个空白编码节点先留空 |
| 随机种子 | 节点 9：`KSampler.seed`；`control_after_generate` 保持 fixed |
| AReA 随机种子 | 节点 3：`AReAFluxPatch.seed`，对照时与采样器种子一致 |
| 分辨率 | 同时改节点 2 `ModelSamplingFlux` 和节点 8 `EmptySD3LatentImage` 的 width / height |
| AReA 开关 | 节点 3：`enabled`，true 开启、false 关闭 |

默认 batch=1、28 步、Euler/simple、KSampler CFG=1、FluxGuidance=4.5、denoise=1、`max_shift=base_shift=1.15`、VAE 分块 512/64。开关对照时保持提示词、两个种子与采样设置一致。公开副本沿用已测连接和数值参数，仅整理模型选择与说明，尚未用公开副本重新出图。

已有原生 Krea 工作流可按 `MODEL → ModelSamplingFlux → AReAFluxPatch → KSampler` 连接。节点保留原采样器和权重。示例固定 shift=1.15，对应上游 FLUX 适配器；换用随分辨率变化的 shift 后，实验条件也随之改变。

## 补丁参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| enabled | true | 开启补丁；false 返回未添加本补丁的模型克隆 |
| phase1_steps | 0 | 自动按实际步数的约 8/28 分配粗阶段；手动值必须为 1 至实际步数减 1 |
| pooling | subsample | 粗阶段 K/V 代表 token：subsample、mean、medoid、random |
| halo | 8 | 窗口邻域最大宽度，单位为 token |
| bridge_ratio | 0.5 | 全局桥接 token 比例；0 关闭桥接 |
| seed | 0 | 窗口抖动和随机池化种子 |
| window_batch_size | 1 | 每次注意力调用处理的窗口数量；调大可能用临时显存换速度 |

自动调度：2 步→1+1，4 步→1+3，8 步→2+6，28 步→8+20。高分辨率单步会报错。阶段按真实 sigma 区间定位，避免把模型重复求值误计为新步数。短步数能运行不代表 Krea-dev 的短步画质已通过，也不代表加速 LoRA 已兼容。

宽高优先使用 32 的倍数。FLUX.1 每 16×16 像素形成一个 image token，高分辨率两轴 token 数需要至少为 2 的公因子。两轴均不超过 64 token 时沿用原始 attention；尺寸不支持时明确报错。

## 修复与限制

- 内容统计、窗口布局和桥接选择逐张计算，避免 batch 内图片相互干扰。批量一致性已经小型随机宿主模型验证，完整权重 batch 验收尚未完成。
- `AReASaveImage` 用时间戳、UUID 与独占创建防止碰撞、并发覆盖；保留 prompt/workflow 元数据，并遵守禁用元数据设置。文件名前缀不能指定子目录，也可改接官方 Save Image。
- 使用模型局部 ComfyUI 补丁接口，不替换全局 attention 或修改权重；冲突的 attention/RoPE/块补丁、参考 token 和自定义 mask 会被拒绝。
- LoRA 组合、GGUF/NF4、torch.compile、多 GPU、ControlNet、IPAdapter 尚未完整验收。尝试普通权重 LoRA 时放在 AReA 前面。
- 尚未实现上游 Bootstrap 模式；这是目标尺寸直接生成，未完成与“低分辨率生成后放大”的比较。
- 全分辨率 QKV 和 MLP 仍有开销，分块 VAE 只降低解码压力；窗口批量、offload 会影响速度，不保证所有 16GB 或更低显存机器都能运行。
- 多题材、多种子、长宽比、编辑场景仍缺验证。ComfyUI attention 接口变化可能需要后续维护适配器。

缺失节点时检查目录层级及启动日志；模型名称无效时重新选择已有文件；显存紧张时先用 2K、batch=1、window_batch_size=1；遇到补丁冲突时先验证随包工作流，再逐项增加扩展。

## 开发与归属

CPU 验证方法见 [TESTING.zh-CN.md](TESTING.zh-CN.md)，反馈方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。GitHub Actions 在推送和拉取请求时执行静态检查与 CPU 回归，当前运行结果见 [Actions](https://github.com/VividMuse-AGI/ComfyUI-AReA/actions)；未配置自动发布。

算法和修改的核心源码来自 [KooroshRH/AReA](https://github.com/KooroshRH/AReA)，原作者为 Javad Rajabi、Koorosh Roohi、David B. Lindell、Babak Taati。参见[项目主页](https://rajabi2001.github.io/area/)、[LICENSE](LICENSE) 和 [NOTICE](NOTICE)。这是独立社区移植，非 AReA 作者或 ComfyUI 团队官方发布。代码采用 Apache-2.0，模型许可单独适用。
