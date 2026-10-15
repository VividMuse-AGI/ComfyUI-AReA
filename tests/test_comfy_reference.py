import os
import unittest

import torch

import test_regressions  # loads the plugin under a stable package name
from comfyui_area.adapter import Config, DiffusionWrapper, MARKER, patch_model, rotate
from comfyui_area.nodes import AReAFluxPatch
from reference_host import install, model_patcher


@unittest.skipUnless(os.environ.get('AREA_COMFY_SOURCE'), 'Set AREA_COMFY_SOURCE to the pinned ComfyUI checkout')
class ComfyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Flux = install(os.environ['AREA_COMFY_SOURCE'])

    def setUp(self):
        torch.manual_seed(11)
        self.model = self.Flux(
            in_channels=16, out_channels=16, vec_in_dim=8, context_in_dim=16,
            hidden_size=32, mlp_ratio=2, num_heads=4, depth=1, depth_single_blocks=1,
            axes_dim=[2, 2, 4], theta=10000, patch_size=2, qkv_bias=True,
            guidance_embed=True, txt_ids_dims=[], operations=torch.nn,
        ).eval()
        self.patcher = model_patcher(self.model)
        self.x = torch.randn(2, 16, 16, 16)
        self.context = torch.randn(2, 3, 16)
        self.y = torch.randn(2, 8)
        self.guidance = torch.full((2,), 3.5)

    def options(self, config=Config(), native=2):
        patched = patch_model(self.patcher, config)
        options = patched.model_options['transformer_options']
        # Same code path, small equivalent grid: 128px is 4x the fixture's native size.
        options['wrappers']['diffusion_model'][MARKER] = [DiffusionWrapper(config, native=native)]
        options['sample_sigmas'] = torch.tensor([1., .8, .5, .2, 0.])
        return options

    def forward(self, options, sigma, batch=None):
        index = slice(None) if batch is None else slice(batch, batch + 1)
        with torch.no_grad():
            return self.model(self.x[index].clone(), torch.full((self.x[index].shape[0],), sigma),
                              self.context[index], y=self.y[index], guidance=self.guidance[index],
                              transformer_options=options)

    def test_both_phases_execute_real_double_and_single_blocks(self):
        options = self.options()
        for sigma in (1., .8, .5, .2):
            output = self.forward(options, sigma)
            self.assertEqual(output.shape, self.x.shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_batched_output_matches_individual_images(self):
        for pooling in ('subsample', 'mean', 'random', 'medoid'):
            options = self.options(Config(pooling=pooling, halo=1, bridge_ratio=.5))
            for sigma in (1., .5):
                together = self.forward(options, sigma)
                separate = torch.cat([self.forward(options, sigma, i) for i in range(2)])
                torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)

    def test_native_resolution_matches_unpatched_model(self):
        baseline = self.forward({}, .5)
        patched = self.forward(self.options(native=64), .5)
        torch.testing.assert_close(baseline, patched, rtol=0, atol=0)

    def test_original_branch_is_unchanged_after_patched_forward(self):
        before = self.forward({}, .5)
        self.forward(self.options(), .5)
        after = self.forward({}, .5)
        self.assertEqual(self.patcher.model_options, {'transformer_options': {}})
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_disabled_node_clones_without_patches(self):
        result, = AReAFluxPatch().patch(self.patcher, enabled=False)
        self.assertIsNot(result, self.patcher)
        self.assertEqual(result.model_options, self.patcher.model_options)

    def test_duplicate_and_conflicting_patches_rejected(self):
        patched = patch_model(self.patcher, Config())
        with self.assertRaises(ValueError):
            patch_model(patched, Config())
        self.patcher.model_options['transformer_options']['optimized_attention_override'] = lambda: None
        with self.assertRaises(ValueError):
            patch_model(self.patcher, Config())

    def test_rotary_math_matches_comfy(self):
        from comfy.ldm.flux.math import apply_rope1
        ids = torch.randn(2, 20, 3)
        pe = self.model.pe_embedder(ids)
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(2, 4, 20, 8).to(dtype)
            torch.testing.assert_close(rotate(x, pe), apply_rope1(x, pe), atol=0, rtol=0)

    def test_exception_does_not_leave_model_mutated(self):
        options = self.options(Config(phase1_steps=4))
        with self.assertRaises(ValueError):
            self.forward(options, 1.)
        self.assertTrue(torch.isfinite(self.forward({}, .5)).all())

    def test_bfloat16_model_executes_both_phases(self):
        self.model.to(dtype=torch.bfloat16)
        self.x, self.context, self.y = [x.bfloat16() for x in (self.x, self.context, self.y)]
        options = self.options()
        for sigma in (1., .5):
            out = self.forward(options, sigma)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out).all())

    def test_conflict_added_after_area_is_rejected(self):
        options = self.options()
        options['patches']['attn1_patch'].append(lambda *args: {})
        with self.assertRaises(ValueError):
            self.forward(options, 1.)


if __name__ == '__main__':
    unittest.main()
