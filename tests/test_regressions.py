import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if "comfyui_area" not in sys.modules:
    spec = importlib.util.spec_from_file_location("comfyui_area", ROOT / "__init__.py",
                                                submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

from comfyui_area.adapter import Config, Runtime, phase1_count, sampling_position, rotate
from comfyui_area.area_core import attention, bridge, schedule, tiling
from comfyui_area import saving

torch.set_num_threads(2)
if os.environ.get("AREA_TEST_TMP"):
    tempfile.tempdir = os.environ["AREA_TEST_TMP"]


class ScheduleTests(unittest.TestCase):
    def test_short_schedules_keep_both_phases(self):
        for steps, coarse in ((2, 1), (4, 1), (8, 2), (12, 3), (28, 8), (50, 14)):
            self.assertEqual(phase1_count(steps), coarse)
            s = schedule.build_schedule(256, 256, steps, coarse, 64)
            self.assertEqual(s.steps, steps)
            self.assertGreater(len(s.factors), 0)
            self.assertGreater(len(s.windows), 0)

    def test_bad_phase_fails_early(self):
        for total, requested in ((1, 0), (4, 4), (8, 8), (4, -1)):
            with self.assertRaises(ValueError):
                phase1_count(total, requested)

    def test_actual_sigmas_not_model_call_count(self):
        sigmas = torch.tensor([1., .85, .6, .3, 0.])
        for t, step in ((1., 0), (.85, 1), (.7, 1), (.6, 2), (.3, 3), (0., 3)):
            for _ in range(3):
                self.assertEqual(sampling_position(sigmas, torch.tensor([t, t])), (step, 4))
        self.assertEqual(sampling_position(sigmas[1:], torch.tensor([.85])), (0, 3))

    def test_invalid_sigmas_rejected(self):
        for sigmas in (None, [1, 0], [1, .5, .5, 0], [1, float("nan"), 0], [1, .5, -1]):
            with self.assertRaises(ValueError):
                sampling_position(sigmas, torch.tensor([1.]))
        with self.assertRaises(ValueError):
            sampling_position([1, .5, 0], torch.tensor([1., .5]))

    def test_pooling_never_reverts_to_full_attention_on_odd_grids(self):
        self.assertEqual(schedule.pooling_ramp(5, 0, 65, 65), [])
        self.assertTrue(all(f >= 2 and 65 % f == 0 for f in schedule.pooling_ramp(5, 8, 65, 65)))

    def test_config_nonfinite_rejected(self):
        for ratio in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                Config(bridge_ratio=ratio)


class KernelTests(unittest.TestCase):
    def test_bridge_batch_cancellation_regression(self):
        a = torch.tensor([0., 1., 3.]).reshape(1, 3, 1, 1)
        together = bridge.token_importance(torch.cat((a, -a)))
        expected = torch.cat((bridge.token_importance(a), bridge.token_importance(-a)))
        torch.testing.assert_close(together, expected)
        self.assertGreater(int(torch.count_nonzero(together)), 0)

    def test_bridge_matches_independent_formula(self):
        k = torch.randn(3, 20, 2, 4, generator=torch.Generator().manual_seed(3))
        expected = (k - k.mean(1, keepdim=True)).square().sum((-1, -2))
        torch.testing.assert_close(bridge.token_importance(k), expected)

    def test_registers_do_not_select_local_tokens(self):
        scores = torch.arange(64).float()
        windows = torch.tensor([[0, 1, 8, 9], [54, 55, 62, 63]])
        regs = bridge.select_registers(scores, windows, 8, 8, 8)
        for w, r in zip(windows, regs):
            self.assertFalse(torch.isin(r, w).any())
            self.assertEqual(len(torch.unique(r)), 8)

    def test_pooling_retains_text_and_same_kv_indices(self):
        k = torch.arange(18.).reshape(1, 18, 1, 1)
        for pooling in attention.POOLING:
            kk, vv = attention.pool_keys_values(k, k + 100, 2, 4, 4, 2, pooling)
            torch.testing.assert_close(kk[:, :2], k[:, :2])
            torch.testing.assert_close(vv - kk, torch.full_like(vv, 100.))
            self.assertEqual(tuple(kk.shape), (1, 6, 1, 1))

    def test_window_attention_matches_dense_mask_oracle(self):
        # Disjoint windows, no halo/bridge: compare to explicit attention per local block.
        gen = torch.Generator().manual_seed(7)
        q, k, v = [torch.randn(1, 18, 2, 4, generator=gen) for _ in range(3)]
        windows = [tiling.Window(r, c, r + 1, c + 1) for r in (0, 2) for c in (0, 2)]
        layout = attention.build_fine_layout(windows, 4, 4, 0, 0, 0, 0, q.dtype, q.device)
        txt, img = attention.windowed_attention(q[:, :2], k[:, :2], v[:, :2],
                                                q[:, 2:], k[:, 2:], v[:, 2:], k[:, 2:], layout)
        expect_img = torch.empty_like(img)
        text_parts = []
        for win in windows:
            idx = tiling.window_tokens(win, 4, q.device)
            all_idx = torch.cat((torch.arange(2), idx + 2))
            out = torch.nn.functional.scaled_dot_product_attention(
                q[:, all_idx].transpose(1, 2), k[:, all_idx].transpose(1, 2),
                v[:, all_idx].transpose(1, 2)).transpose(1, 2)
            expect_img[:, idx] = out[:, 2:]
            text_parts.append(out[:, :2])
        torch.testing.assert_close(img, expect_img)
        torch.testing.assert_close(txt, torch.stack(text_parts).mean(0))

    def test_window_chunking_does_not_change_results(self):
        gen = torch.Generator().manual_seed(10)
        q, k, v = [torch.randn(1, 66, 2, 4, generator=gen) for _ in range(3)]
        windows = [tiling.Window(r, c, r + 3, c + 3) for r in (0, 4) for c in (0, 4)]
        layout = attention.build_fine_layout(windows, 8, 8, 1, 4, 8, 5, q.dtype, q.device)
        args = (q[:, :2], k[:, :2], v[:, :2], q[:, 2:], k[:, 2:], v[:, 2:], k[:, 2:], layout)
        one = attention.windowed_attention(*args, window_batch_size=1)
        many = attention.windowed_attention(*args, window_batch_size=8)
        for a, b in zip(one, many):
            torch.testing.assert_close(a, b)

    def test_adaptive_tiling_has_full_coverage_and_budget(self):
        rng = np.random.default_rng(17)
        for h, w, size in ((16, 16, 11), (16, 8, 7), (12, 18, 8)):
            energy = rng.random((h, w)).astype(np.float32)
            wins = tiling.adaptive_tiling(energy, h, w, size, 4, 2, tiling.step_rng(5, 3, 1))
            layout = attention.build_fine_layout(wins, h, w, 1, 4, 5, 3, torch.float32, 'cpu')
            self.assertTrue((layout.coverage > 0).all())
            weights = torch.zeros(h * w)
            for group in layout.groups:
                weights.index_add_(0, group.tokens.flatten(), group.weights)
            torch.testing.assert_close(weights / layout.coverage, torch.ones_like(weights))


class SavingTests(unittest.TestCase):
    def test_chinese_and_workflow_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = {"nodes": [{"type": "AReAFluxPatch", "seed": 5}]}
            name = saving.save_png(np.zeros((4, 5, 3)), directory, "森林里的女孩",
                                   {"text": "中文提示词"}, {"workflow": workflow})
            self.assertTrue(name.startswith("森林里的女孩_"))
            with Image.open(Path(directory) / name) as image:
                self.assertEqual(image.size, (5, 4))
                self.assertEqual(json.loads(image.info["workflow"]), workflow)
                self.assertEqual(json.loads(image.info["prompt"])["text"], "中文提示词")

    def test_parallel_saves_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            def save(i):
                return saving.save_png(np.full((4, 4, 3), i / 16), directory, "同名提示词")
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                names = list(pool.map(save, range(16)))
            self.assertEqual(len(set(names)), 16)
            self.assertEqual(len(list(Path(directory).glob('*.png'))), 16)

    def test_exclusive_create_survives_forced_collision(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(saving, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = 'fixed'
            with patch.object(saving.uuid, 'uuid4') as uid:
                uid.return_value.hex = 'same'
                first = saving.save_png(np.zeros((2, 2, 3)), directory)
                before = (Path(directory) / first).read_bytes()
                uid.side_effect = [type('ID', (), {'hex': 'same'})(), type('ID', (), {'hex': 'new'})()]
                second = saving.save_png(np.ones((2, 2, 3)), directory)
            self.assertNotEqual(first, second)
            self.assertEqual((Path(directory) / first).read_bytes(), before)

    def test_prefix_cannot_escape_output_and_metadata_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            name = saving.save_png(np.zeros((2, 2, 3)), directory, '../escape\\file',
                                   {"private": True}, metadata_enabled=False)
            self.assertEqual(Path(name).name, name)
            with Image.open(Path(directory) / name) as image:
                self.assertNotIn('prompt', image.info)


if __name__ == '__main__':
    unittest.main()
