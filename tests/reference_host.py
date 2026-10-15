"""CPU fixture loading actual ComfyUI FLUX code from AREA_COMFY_SOURCE.

Only host services (GPU management and kernel selection) are stubbed. The FLUX
model, blocks, rotary functions, patch wrapper, and PyTorch attention wrapper
execute the referenced ComfyUI source. No checkpoint or CUDA is needed.
Run in a separate test process, not inside a running ComfyUI server.
"""

import ast
import functools
import importlib.util
from pathlib import Path
import sys
import types

import torch


def install(source):
    source = Path(source)

    def module(name):
        value = types.ModuleType(name)
        value.__path__ = []
        sys.modules[name] = value
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(sys.modules[parent], child, value)
        return value

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, source / path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        parent, child = name.rsplit('.', 1)
        setattr(sys.modules[parent], child, value)
        spec.loader.exec_module(value)
        return value

    for name in ('comfy', 'comfy.ldm', 'comfy.ldm.modules', 'comfy.ldm.flux'):
        module(name)
    mm = module('comfy.model_management')
    mm.in_training = True  # exercise the CPU torch path instead of optional CUDA kernels
    mm.supports_fp64 = lambda device: True
    module('comfy.quant_ops')
    rms = module('comfy.rmsnorm')
    rms.rms_norm = torch.nn.functional.rms_norm
    ops = module('comfy.ops')
    ops.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    attn = module('comfy.ldm.modules.attention')
    attn.__dict__.update(torch=torch, functools=functools, comfy=sys.modules['comfy'], SDP_BATCH_LIMIT=32768)
    source_ast = ast.parse((source / 'comfy/ldm/modules/attention.py').read_text(encoding='utf-8'))
    names = {'AttentionTensorContainer', 'wrap_attn', 'attention_pytorch'}
    selected = ast.Module(body=[n for n in source_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                               and n.name in names], type_ignores=[])
    exec(compile(selected, str(source / 'comfy/ldm/modules/attention.py'), 'exec'), attn.__dict__)
    attn.optimized_attention = attn.attention_pytorch
    attn.ComfyAttention = type('ComfyAttention', (), {'function': None})
    load('comfy.patcher_extension', 'comfy/patcher_extension.py')
    load('comfy.ldm.common_dit', 'comfy/ldm/common_dit.py')
    load('comfy.ldm.flux.math', 'comfy/ldm/flux/math.py')
    load('comfy.ldm.flux.layers', 'comfy/ldm/flux/layers.py')
    return load('comfy.ldm.flux.model', 'comfy/ldm/flux/model.py').Flux


def model_patcher(diffusion):
    from comfy.patcher_extension import copy_nested_dicts

    class Patcher:
        def __init__(self):
            self.model_options = {'transformer_options': {}}

        def get_model_object(self, name):
            assert name == 'diffusion_model'
            return diffusion

        def clone(self):
            other = Patcher()
            other.model_options = copy_nested_dicts(self.model_options)
            return other

        def set_model_patch(self, value, name):
            self.model_options['transformer_options'].setdefault('patches', {}).setdefault(name, []).append(value)

        def set_model_post_input_patch(self, value):
            self.set_model_patch(value, 'post_input')

        def set_model_attn1_patch(self, value):
            self.set_model_patch(value, 'attn1_patch')

    return Patcher()
