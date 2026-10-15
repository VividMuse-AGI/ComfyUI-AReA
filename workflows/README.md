# Annotated workflows / 带注释工作流

English Notes are in this folder; Chinese Notes are in `zh-CN/`. Each folder contains the same three graphs and matching API prompts. Node names stay at their defaults in both languages. The plugin's own node display names currently use Chinese text.

本目录为英文便签，`zh-CN/` 为中文便签。两种语言的连接和执行参数一致，均保留默认节点名称；插件自身的节点显示文字目前为中文。

| UI file, in either folder | Size | `enabled` |
|---|---|---|
| `01_Krea_FP8_2K_AReA_ON.json` | 2048×2048 | true |
| `02_Krea_FP8_4K_AReA_ON.json` | 4096×4096 | true |
| `03_Krea_FP8_4K_AReA_OFF.json` | 4096×4096 | false |

Drag the UI JSON into ComfyUI. A same-stem `_API.json` is an executable API prompt, not the visual graph. It is provided for inspection and later submission; extracting or verifying this package submits nothing.

拖入 UI JSON；同名 `_API.json` 是供程序提交的执行格式。解压或核验本包不会提交任务。

After import, reselect your installed diffusion weight, both CLIP encoders, and VAE. The public graph uses model basenames, so weights in subfolders must be selected again. It includes no weights or model download URLs in the graph metadata.

导入后重新选择扩散模型、两个 CLIP 编码器与 VAE。公开版本使用文件名，不绑定目录；权重在子文件夹时需重新选择。工作流不附带权重或元数据下载地址。

| Colored group / 分组 | Node / 节点 | Control / 修改项 |
|---|---|---|
| 1. Prompt / 提示词 | 5, CLIPTextEncode | Populated text / 有内容的文本框 |
| 2. Seed / 种子 | 9, KSampler | seed; control_after_generate=fixed |
| 3. Resolution A / 分辨率 A | 2, ModelSamplingFlux | width, height |
| 3. Resolution B / 分辨率 B | 8, EmptySD3LatentImage | width, height; match node 2 / 与节点 2 一致 |
| 4. AReA switch / 开关 | 3, AReAFluxPatch | enabled; seed matches KSampler / 种子与采样器一致 |

28 steps, Euler/simple, CFG=1, guidance=4.5, denoise=1, fixed shift=1.15, batch=1. Automatic phase split is 8+20. VAE is tiled at 512 with overlap 64. Start at 2K; public copies were checked statically and have not been used for a fresh generation test.

默认 28 步、Euler/simple、CFG=1、guidance=4.5、denoise=1、固定 shift=1.15、batch=1、自动阶段 8+20、VAE 分块 512/64。建议先用 2K；公开副本完成静态检查，未重新出图。

The 4K ON/OFF prompts differ only in `AReAFluxPatch.enabled` and the output filename prefix. The two 4K UI files also differ in their explanatory text and graph identity. See [English instructions](../README.md) or [中文使用说明](../README.zh-CN.md).
