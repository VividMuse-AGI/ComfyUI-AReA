# Changelog / 更新记录

## 0.1 — experimental prerelease, 2026-10-15

First public version number; 0.1.1 and 0.1.2 below refer to earlier internal local builds, not public releases. This public version is designated experimental through release status and documentation.

首次对外版本号为 0.1；下方的 0.1.1、0.1.2 是此前本机开发编号，不是已发布版本。实验性状态通过发布标记和文档说明。

- Added English and Chinese installation, usage, limitation and testing documentation.
- Prepared 2K ON / 4K ON / 4K OFF workflows with English or Chinese Notes and matching API prompts; default node names and tested numeric settings preserved.
- Replaced machine-specific model subfolders with basenames and documented reselection after import; removed frontend-extension metadata.
- Normalized local paths in historical evidence; added source hashes, CSV metrics, and later browser/annotation verification records.
- Added static packaging verification, CPU CI definition and an issue-report template. Current remote CI results are available in the repository Actions tab.
- Production and existing test Python files remain byte-identical to tested 0.1.1 / packaged 0.1.2. No new GPU generation was performed.
- Final local audit removed stale named-widget extension fields, completed source-file attribution, and clarified that image observations were assistant inspections without independent human ratings. Public version remains 0.1.
- Shortened canvas Notes to one or two lines; detailed parameters and test results remain in the documentation.

补齐双语说明、六份注释工作流及配套 API、公开测试记录、路径整理、静态核验与 CPU CI 定义。推理及既有测试代码保持原哈希，无新 GPU 出图；本版为实验性预发布。

## Internal local build 0.1.2 — 2026-10-15

Packaged native ComfyUI FLUX.1 integration, portable installation material, workflow templates and recorded test evidence. Completed 4/6 P0 images and 0/12 P1 images; remaining tests stopped at user request. Two completed repeated pairs showed about 1.81× sampling speedup without stable whole-card VRAM reduction.

## Internal local build 0.1.1 and earlier — 2026-10-15

Implemented the model patch and exclusive PNG saving; fixed actual-step phase allocation, independent per-image bridge/layout decisions and output overwrite handling. Historical CPU/reference regressions, actual-host CPU/CUDA fixtures, 2K generation and 4K comparisons are documented in TESTING.md. These entries describe local development history, not previously published GitHub tags.
