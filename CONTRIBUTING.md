# Contributing / 反馈与开发

This is an experimental independent port. English and Chinese reports are welcome. Start with the included 2K graph and report the smallest reproducible issue.

Please include the ComfyUI version/commit, operating system, GPU/VRAM, Python/torch version, exact model format, AReA parameters, sampler, dimensions, steps, and other active attention/control patches. Attach a workflow and the relevant error log after removing local paths and credentials. Distinguish a missing node/import error from model-selection failure, out-of-memory, patch conflict or image artifact.

For image-quality reports, include ON/OFF results with the same prompt, both fixed seeds and sampling settings, where practical. State changes in identity/composition rather than assuming same-seed subject identity. Report timing and memory measurement boundaries and avoid treating whole-card NVML usage as node-exclusive memory. Do not send model weights.

Run `python tools/verify_release.py` and the CPU tests described in [TESTING.md](TESTING.md). Inference changes need relevant regression coverage; documentation/workflow corrections need static and UI/API consistency checks. Preserve upstream attribution and identify modified upstream source in NOTICE. If code changes, update the recorded inference baseline deliberately rather than silently relabeling historical GPU results as validation of new code.

欢迎中英文反馈。先从随包 2K 工作流复现，提供 ComfyUI 版本/提交、系统、显卡/显存、Python/torch、模型格式、补丁参数、采样设置与其他注意力扩展。附清理本机路径及凭据后的工作流和相关报错；不要上传权重。

画质反馈尽量提供同提示词、双种子固定、相同参数的开关对照，注明身份/构图变化；耗时和显存写清测量范围。推理代码修改应补相关回归；文档和工作流修改检查静态结构及 UI/API 一致性。保留上游归属，不把历史 GPU 记录重新标成新代码的验收结果。
