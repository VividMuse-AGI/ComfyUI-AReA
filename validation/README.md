# Evidence index / 测试证据索引

These files distinguish historical measurements from current package checks. Local paths and service addresses are normalized for public distribution. A JSON pass value is a recorded historical outcome, not a new execution.

下列文件区分历史测量与本次打包检查；公开副本清理了本机路径和服务地址。历史 JSON 的通过状态不代表本次重新执行。

| File | Scope / 范围 |
|---|---|
| [4k28_repeat.json](4k28_repeat.json) | Four successful repeated 4K runs, metrics, model/code/output hashes; P0 4/6, P1 0/12 / 四次重复实测与完成范围 |
| [4k28_runs.csv](4k28_runs.csv) | Derived per-run metrics, successful repeat runs only / 四次成功运行的指标提取 |
| [4k28_initial.json](4k28_initial.json) | Earlier single ON/OFF pair; separate protocol / 初次单对记录，单独统计 |
| [4k_comparison.jpg](4k_comparison.jpg) | Existing downscaled comparison, OFF left and ON right / 已有缩小对照预览 |
| [gpu_acceptance.json](gpu_acceptance.json) | Historical 1280/2 smoke and 2048/28 generation / 历史冒烟与 2K 出图 |
| [gpu_4k_smoke.json](gpu_4k_smoke.json) | Historical 4096/2 execution smoke / 历史 4K 两步冒烟 |
| [png_validation.json](png_validation.json) | Saved dimensions and metadata; local image paths normalized / 图片尺寸与元数据，路径已整理 |
| [local_host_validation.json](local_host_validation.json), [CUDA](local_host_validation_cuda.json) | Historical real ModelPatcher fixtures / 历史真实宿主接口小模型验证 |
| [historical-regressions.txt](historical-regressions.txt), [local-regressions.txt](local-regressions.txt) | Preserved historical 27-test logs / 历史 27 项回归记录 |
| [frontend_history.json](frontend_history.json) | Historical browser load, 2K annotations, 4K file checks / 历史浏览器及注释验证 |
| [workflow_portability.json](workflow_portability.json) | Six public annotated copies, source hashes and parameter preservation / 六份公开注释副本来源与参数保持 |
| [historical_sources.json](historical_sources.json) | Source/normalized record SHA256 and transformations / 原记录和整理后记录哈希及转换说明 |
| [inference_baseline.json](inference_baseline.json) | Byte-identical production/test Python baselines / 推理和既有测试代码一致性 |
| [prerelease_cpu_regressions.txt](prerelease_cpu_regressions.txt), [environment](prerelease_cpu_environment.json) | CPU-only local checks for this package / 本次独立 CPU 检查 |
| [packaging_verification.json](packaging_verification.json) | Static local release verification / 本次静态发布核验 |
| [license_audit.json](license_audit.json) | Pinned upstream license blob, author attribution and source-file notices / 固定上游许可证与归属核对 |
| [final_audit.json](final_audit.json), [中文审核说明](../FINAL_AUDIT.zh-CN.md) | Final local prerelease audit and corrected findings / 本次最终审核及修正项 |

Full original PNGs, continuous telemetry, process logs and machine-specific benchmark runners are not bundled. Historical output hashes permit comparison with the retained local archive; the source hashes do not imply those original files are available for download. The comparison JPEG is a visual preview, not a substitute for original 4K inspection. Visual observations were made by the development assistant, without an independent human rating study.

未打包原生大图、连续遥测、进程日志和依赖本机目录的基准执行器。输出哈希可用于与保留的本地归档核对；原记录哈希不代表原文件已公开下载。JPEG 预览不能替代原生 4K 局部检查。
