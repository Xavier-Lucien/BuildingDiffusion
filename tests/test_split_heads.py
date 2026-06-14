"""校验 split-heads 路径：bbox 扩散 + class CE + objectness BCE。"""
import os
import unittest

import numpy as np
import torch

from _helpers import (
    tiny_net_config, tiny_data_config, write_stats_file, make_tempdir,
    N_CTX, CLASS_DIM,
)
from model.builder import build_model
from dataset.encoding import post_process

N_CLASSES = 15
N_MATERIALS = CLASS_DIM - 1  # 13


def _batch(B=2, N=N_CTX, n_real=5):
    """class_labels 用 {-1,+1}：前 n_real 个真实槽(end=-1)，其余空槽(end=+1)。"""
    cls = -torch.ones(B, N, CLASS_DIM)
    for b in range(B):
        for i in range(N):
            if i < n_real:
                m = (i * 2 + b) % N_MATERIALS
                cls[b, i, m] = 1.0      # 材质 one-hot
                cls[b, i, -1] = -1.0    # end=-1 -> 真实
            else:
                cls[b, i, -1] = 1.0     # end=+1 -> 空槽
    return {
        "translations": torch.rand(B, N, 3) * 2 - 1,
        "sizes": torch.rand(B, N, 3) * 2 - 1,
        "angles": torch.rand(B, N, 2) * 2 - 1,
        "class_labels": cls,
    }


def _split_config(loss_iou=False, stats=None):
    net = tiny_net_config(loss_iou=loss_iou, split_heads=True)
    if stats is not None:
        net["diffusion_kwargs"]["train_stats_file"] = stats
    return {"network": net, "data": tiny_data_config()}


class TestSplitForward(unittest.TestCase):
    def test_forward_scalar_loss_with_ce_bce(self):
        model = build_model(_split_config(loss_iou=False), N_CLASSES, device="cpu")
        out = model(_batch())
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertIn("loss.class", out)
        self.assertIn("loss.obj", out)
        out["loss"].backward()  # 可反向

    def test_forward_with_iou(self):
        stats = os.path.join(make_tempdir(), "dataset_stats.txt")
        write_stats_file(stats)
        model = build_model(_split_config(loss_iou=True, stats=stats), N_CLASSES, device="cpu")
        out = model(_batch())
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_denoiser_three_heads_shapes(self):
        model = build_model(_split_config(), N_CLASSES, device="cpu")
        self.assertTrue(model.denoiser.split_heads)
        B, N = 2, N_CTX
        bbox = torch.rand(B, N, 8) * 2 - 1
        t = torch.zeros(B, dtype=torch.int64)
        out_bbox, class_logits, obj_logit = model.denoiser(bbox, t, None, None)
        self.assertEqual(tuple(out_bbox.shape), (B, N, 8))
        self.assertEqual(tuple(class_logits.shape), (B, N, N_MATERIALS))
        self.assertEqual(tuple(obj_logit.shape), (B, N, 1))


class TestSplitSampling(unittest.TestCase):
    def test_generate_filters_and_post_process(self):
        model = build_model(_split_config(), N_CLASSES, device="cpu")
        model.eval()
        samples = model.generate_layout(
            batch_size=1, num_points=N_CTX, point_dim=22, device="cpu", clip_denoised=True
        )
        self.assertEqual(len(samples), 1)
        raw = samples[0]
        # objectness 过滤后 K<=N；class_labels 为 13 维材质 logit
        self.assertEqual(raw["class_labels"].shape[-1], N_MATERIALS)
        self.assertLessEqual(raw["translations"].shape[1], N_CTX)

        bounds = {
            "translations": (np.zeros(3, np.float32), np.ones(3, np.float32)),
            "sizes": (np.zeros(3, np.float32), np.ones(3, np.float32)),
            "angles": (-np.pi, np.pi),
        }
        processed = post_process(raw, bounds)
        self.assertEqual(processed["angles"].shape[-1], 1)


class TestDefaultPathUnaffected(unittest.TestCase):
    def test_non_split_still_works(self):
        cfg = {"network": tiny_net_config(loss_iou=False, split_heads=False),
               "data": tiny_data_config()}
        model = build_model(cfg, N_CLASSES, device="cpu")
        self.assertFalse(model.split_heads)
        out = model(_batch())
        self.assertEqual(out["loss"].dim(), 0)
        self.assertNotIn("loss.obj", out)  # 旧路径无 objectness


if __name__ == "__main__":
    unittest.main()
