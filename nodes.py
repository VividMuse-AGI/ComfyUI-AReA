from .adapter import Config, patch_model


class AReAFluxPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "phase1_steps": ("INT", {"default": 0, "min": 0, "max": 10000,
                                    "tooltip": "0 自动按实际步数分配粗阶段（28 步时为 8+20）；手动值必须小于实际步数。"}),
            "pooling": (["subsample", "mean", "medoid", "random"],),
            "halo": ("INT", {"default": 8, "min": 0, "max": 64}),
            "bridge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False,
                             "tooltip": "AReA 窗口抖动/随机池化种子，建议与采样器种子一致。"}),
            "window_batch_size": ("INT", {"default": 1, "min": 1, "max": 64,
                                          "tooltip": "每次注意力计算的窗口数量。1 更省显存，增加可能加快速度。"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "AReA/FLUX"
    DESCRIPTION = "FLUX.1-Krea-dev / FLUX.1-dev 高分辨率注意力补丁；不修改原模型权重或采样器。"

    def patch(self, model, enabled=True, phase1_steps=0, pooling="subsample", halo=8,
              bridge_ratio=0.5, seed=0, window_batch_size=1):
        if not enabled:
            return (model.clone(),)
        config = Config(phase1_steps, pooling, halo, bridge_ratio, seed, window_batch_size)
        return (patch_model(model, config),)


class AReASaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "AReA_Krea"})},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "AReA/image"
    DESCRIPTION = "使用独占创建保存 PNG，避免不同提示词、参数或并发任务覆盖；保留工作流元数据。"

    def save_images(self, images, filename_prefix="AReA_Krea", prompt=None, extra_pnginfo=None):
        import folder_paths
        from comfy.cli_args import args
        from .saving import save_png

        results = []
        for image in images:
            filename = save_png(image.detach().cpu().float().numpy(), folder_paths.get_output_directory(),
                                filename_prefix, prompt, extra_pnginfo,
                                metadata_enabled=not args.disable_metadata)
            results.append({"filename": filename, "subfolder": "", "type": "output"})
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {"AReAFluxPatch": AReAFluxPatch, "AReASaveImage": AReASaveImage}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AReAFluxPatch": "AReA · FLUX.1 / Krea 高分辨率补丁",
    "AReASaveImage": "AReA · 保存图像（防覆盖）",
}
