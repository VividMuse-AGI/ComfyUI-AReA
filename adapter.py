"""Per-forward AReA integration using ComfyUI patch APIs, without global monkey-patches."""

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from .area_core import attention, tiling
from .area_core.schedule import build_schedule

KEY = "area_flux_runtime_v1"
MARKER = "area_flux_config_v1"


@dataclass(frozen=True)
class Config:
    phase1_steps: int = 0  # 0: scale the original 8/28 split to the actual schedule
    pooling: str = "subsample"
    halo: int = 8
    bridge_ratio: float = 0.5
    seed: int = 0
    window_batch_size: int = 1

    def __post_init__(self):
        if self.pooling not in attention.POOLING:
            raise ValueError("Unknown AReA pooling method")
        if not 0 <= self.phase1_steps <= 10000 or not 0 <= self.halo <= 64:
            raise ValueError("Invalid AReA phase1_steps or halo")
        if not math.isfinite(self.bridge_ratio) or not 0 <= self.bridge_ratio <= 1:
            raise ValueError("AReA bridge_ratio must be between 0 and 1")
        if not 1 <= self.window_batch_size <= 64 or not 0 <= self.seed < 2**64:
            raise ValueError("Invalid AReA window batch size or seed")


def phase1_count(total, requested=0):
    if total < 2:
        raise ValueError("AReA 高分辨率生成至少需要 2 步；建议 Krea-dev 从 28 步开始。")
    if requested < 0:
        raise ValueError("phase1_steps must be non-negative")
    if requested >= total:
        raise ValueError(
            f"AReA 粗阶段 {requested} 步不能大于等于实际采样 {total} 步。"
            "请将 phase1_steps 设为 0（自动），或设为 1 到实际步数减 1。"
        )
    return requested or max(1, min(total - 1, int(total * 8 / 28 + 0.5)))


def sampling_position(sigmas, timestep):
    """Use sigma intervals, not forward-call counts (Heun/repeated CFG calls are safe)."""
    if sigmas is None:
        raise ValueError("AReA 需要采样器提供 sample_sigmas；请更新 ComfyUI 并使用标准采样节点。")
    s = torch.as_tensor(sigmas).detach().to(device="cpu", dtype=torch.float64).flatten()
    if len(s) < 3 or not torch.isfinite(s).all() or (s < 0).any() or not (s[:-1] > s[1:]).all():
        raise ValueError("AReA 需要至少 2 步、有限且严格递减的 sigma 序列。")
    values = torch.as_tensor(timestep).detach().to(device="cpu", dtype=torch.float64).flatten()
    if not torch.isfinite(values).all() or not torch.allclose(values, values[:1].expand_as(values)):
        raise ValueError("AReA does not support mixed timesteps within one model batch")
    current = values[0]
    # Small float conversion differences at exact boundaries must not select a prior step.
    distance = (s[:-1] - current).abs()
    nearest = int(distance.argmin())
    if float(distance[nearest]) < 1e-6:
        index = nearest
    else:
        index = int(torch.searchsorted(-s, -current, right=True)) - 1
    return max(0, min(len(s) - 2, index)), len(s) - 1


@lru_cache(maxsize=128)
def scheduled(h, w, steps, phase1, native, txt_len):
    return build_schedule(h, w, steps, phase1, native, txt_len=txt_len)


def rotate(x, pe):
    # Import at the host boundary so standalone kernel tests need no ComfyUI installation.
    from comfy.ldm.flux.math import apply_rope1
    return apply_rope1(x, pe)


class Runtime:
    """Lives only inside one diffusion forward, never on a shared model or patch object."""

    def __init__(self, model, config, h, w, step, total, native=64):
        self.model, self.config = model, config
        self.h, self.w, self.step, self.total, self.native = h, w, step, total, native
        self.phase = None
        self.pe = None
        self.layouts = []

    def prepare(self, data):
        img, txt = data["img"], data["txt"]
        self.txt_len = int(txt.shape[1])
        if img.shape[1] != self.h * self.w:
            raise ValueError("AReA currently supports a single FLUX.1 image grid, without reference tokens")
        plan = scheduled(self.h, self.w, self.total,
                         phase1_count(self.total, self.config.phase1_steps), self.native, self.txt_len)
        self.phase, self.size = plan.at(self.step)
        if self.phase == "coarse":
            f = self.size
            ids = data["img_ids"].clone()
            ids[..., 1:] = torch.div(ids[..., 1:], f, rounding_mode="floor")
            kv_ids = ids.reshape(ids.shape[0], self.h, self.w, -1)[:, f // 2::f, f // 2::f]
            kv_ids = kv_ids.reshape(ids.shape[0], -1, ids.shape[-1])
            self.rope_q = self.model.pe_embedder(torch.cat((data["txt_ids"], ids), dim=1))
            self.rope_k = self.model.pe_embedder(torch.cat((data["txt_ids"], kv_ids), dim=1))
        else:
            for sample in img.split(1):
                energy = tiling.detail_map(sample, self.h, self.w)
                windows = tiling.adaptive_tiling(
                    energy, self.h, self.w, self.size, min(self.native, max(self.h, self.w)),
                    self.txt_len, rng=tiling.step_rng(self.config.seed, self.step, 0x5EA1),
                )
                registers = max(1, round(self.size * self.config.bridge_ratio)) ** 2 if self.config.bridge_ratio else 0
                registers, halo = attention.fit_extra_keys(
                    windows, self.h, self.w, self.txt_len, registers, self.config.halo)
                self.layouts.append(attention.build_fine_layout(
                    windows, self.h, self.w, halo, registers, self.config.seed,
                    self.step, sample.dtype, sample.device))
        return data

    def attend(self, original, q, k, v, heads, kwargs):
        if self.phase is None or self.pe is None:
            raise RuntimeError("AReA hooks did not run; this ComfyUI attention path is unsupported")
        if kwargs.get("mask") is not None:
            raise ValueError("AReA does not yet support custom attention masks")
        options = kwargs.get("transformer_options", {})
        block = int(options.get("block_index", 0))
        if options.get("block_type") == "single":
            block += len(self.model.double_blocks)

        def backend(qq, kk, vv, mask=None):
            call_kwargs = dict(kwargs)
            # Call the original backend, without recursively entering this override.
            call_kwargs.pop("transformer_options", None)
            call_kwargs.update(skip_reshape=True, skip_output_reshape=False, mask=mask,
                               _inside_attn_wrapper=True)
            result = original(qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
                              heads, **call_kwargs)
            return result.reshape(qq.shape[0], qq.shape[1], heads, qq.shape[-1])

        if self.phase == "coarse":
            key, value = attention.pool_keys_values(
                k.transpose(1, 2), v.transpose(1, 2), self.txt_len, self.h, self.w,
                self.size, self.config.pooling, seed=self.config.seed, step=self.step, block=block)
            result = backend(rotate(q, self.rope_q).transpose(1, 2),
                             rotate(key.transpose(1, 2), self.rope_k).transpose(1, 2), value)
        else:
            qr, kr = rotate(q, self.pe).transpose(1, 2), rotate(k, self.pe).transpose(1, 2)
            vv, ku = v.transpose(1, 2), k.transpose(1, 2)
            results = []
            t = self.txt_len
            if len(self.layouts) != q.shape[0]:
                raise ValueError("AReA batch/layout mismatch")
            for b, layout in enumerate(self.layouts):
                # Separate every content-dependent operation across images/CFG branches.
                text, img = attention.windowed_attention(
                    qr[b:b+1, :t], kr[b:b+1, :t], vv[b:b+1, :t],
                    qr[b:b+1, t:], kr[b:b+1, t:], vv[b:b+1, t:], ku[b:b+1, t:],
                    layout, backend=backend, window_batch_size=self.config.window_batch_size)
                results.append(torch.cat((text, img), dim=1))
            result = torch.cat(results, dim=0)
        if kwargs.get("skip_output_reshape", False):
            return result.transpose(1, 2)
        return result.flatten(2)


def post_input(data):
    runtime = data["transformer_options"].get(KEY)
    return runtime.prepare(data) if runtime is not None else data


def pre_attention(q, k, v, pe=None, attn_mask=None, extra_options=None):
    runtime = (extra_options or {}).get(KEY)
    if runtime is None:
        return {}
    if attn_mask is not None:
        raise ValueError("AReA currently supports unmasked FLUX.1 text-to-image attention")
    runtime.pe = pe
    # AReA applies the correct Q and K embeddings after pooling them separately.
    return {"pe": None}


def attention_override(original, q, k, v, heads, **kwargs):
    runtime = kwargs.get("transformer_options", {}).get(KEY)
    if runtime is None:
        return original(q, k, v, heads, **kwargs)
    return runtime.attend(original, q, k, v, heads, kwargs)


@dataclass(frozen=True)
class DiffusionWrapper:
    config: Config
    native: int = 64

    def __call__(self, executor, x, timestep, context, y=None, guidance=None,
                 ref_latents=None, control=None, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        h_lat, w_lat = x.shape[-2:]
        h, w = (h_lat + 1) // 2, (w_lat + 1) // 2
        if max(h, w) > self.native:
            patches = options.get("patches", {})
            if (options.get("optimized_attention_override") is not attention_override
                    or patches.get("post_input") != [post_input]
                    or patches.get("attn1_patch") != [pre_attention]
                    or options.get("patches_replace")):
                raise ValueError("AReA 检测到后续添加的冲突注意力补丁；请分开模型分支使用。")
            if ref_latents is not None or options.get("rope_options"):
                raise ValueError("AReA cannot be combined with reference latents or other RoPE patches")
            if h_lat % 2 or w_lat % 2 or math.gcd(h, w) < 2:
                raise ValueError("AReA 尺寸不兼容：请将宽高设为 32 的倍数，例如 2048×2048 或 4096×2304。")
            step, total = sampling_position(options.get("sample_sigmas"), timestep)
            phase1_count(total, self.config.phase1_steps)  # fail before any transformer layers
            options[KEY] = Runtime(executor.class_obj, self.config, h, w, step, total, self.native)
        return executor(x, timestep, context, y, guidance, ref_latents, control, options, **kwargs)


def patch_model(model, config):
    from comfy.patcher_extension import WrappersMP, add_wrapper_with_key

    diffusion = model.get_model_object("diffusion_model")
    p = getattr(diffusion, "params", None)
    if (type(diffusion).__name__ != "Flux" or p is None or p.in_channels != 16
            or p.patch_size != 2 or not p.guidance_embed or getattr(p, "global_modulation", False)):
        raise ValueError("此节点支持 ComfyUI 原生 FLUX.1-dev / FLUX.1-Krea-dev，请使用对应模型加载器。")
    if not hasattr(model, "set_model_post_input_patch"):
        raise RuntimeError("ComfyUI 版本过旧：缺少 FLUX post_input patch 接口，请更新。")
    options = model.model_options.get("transformer_options", {})
    if MARKER in options:
        raise ValueError("模型已经应用 AReA，请只保留一个 AReA Patch 节点。")
    patches = options.get("patches", {})
    if (options.get("optimized_attention_override") or options.get("rope_options")
            or options.get("patches_replace") or any(patches.get(k) for k in ("attn1_patch", "post_input"))):
        raise ValueError("AReA 与已有注意力/RoPE/块替换补丁冲突；请将它们分成独立模型分支测试。")
    result = model.clone()
    result.set_model_post_input_patch(post_input)
    result.set_model_attn1_patch(pre_attention)
    opts = result.model_options.setdefault("transformer_options", {})
    opts[MARKER] = config
    opts["optimized_attention_override"] = attention_override
    add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, MARKER, DiffusionWrapper(config), opts)
    return result
