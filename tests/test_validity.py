"""校验 size 越界惩罚 / angle 单位化正则，以及 post_process 的 size 兜底。"""
import unittest

import numpy as np
import torch

from _helpers import tiny_net_config, tiny_data_config, N_CTX, CLASS_DIM
from model.builder import build_model
from dataset.encoding import post_process

N_CLASSES = 15


def _model(split_heads=False, loss_validity=True):
    cfg = {
        "network": tiny_net_config(loss_iou=False, split_heads=split_heads,
                                  loss_validity=loss_validity),
        "data": tiny_data_config(),
    }
    return build_model(cfg, N_CLASSES, device="cpu")


class TestValidityLoss(unittest.TestCase):
    def setUp(self):
        self.diff = _model().diffusion  # 直接拿 GaussianDiffusion 测辅助函数

    def test_size_penalty_positive_when_out_of_bounds(self):
        x0 = torch.zeros(1, 4, 8)
        x0[:, :, 3:6] = 2.0   # size 超出 [-1,1]
        x0[:, :, 6] = 1.0     # cos
        size_valid, angle_norm = self.diff._validity_loss(x0)
        self.assertGreater(float(size_valid.mean()), 0.5)
        self.assertAlmostEqual(float(angle_norm.mean()), 0.0, places=5)

    def test_size_penalty_zero_in_bounds(self):
        x0 = torch.zeros(1, 4, 8)
        x0[:, :, 3:6] = 0.3   # size 合法
        x0[:, :, 6] = 1.0
        size_valid, _ = self.diff._validity_loss(x0)
        self.assertAlmostEqual(float(size_valid.mean()), 0.0, places=5)

    def test_angle_norm_penalty(self):
        x0 = torch.zeros(1, 4, 8)
        x0[:, :, 6] = 2.0     # cos=2, sin=0 -> 模长 2
        _, angle_norm = self.diff._validity_loss(x0)
        self.assertAlmostEqual(float(angle_norm.mean()), 1.0, places=4)  # (2-1)^2


class TestValidityForward(unittest.TestCase):
    def _batch(self, B=2, N=N_CTX):
        return {
            "translations": torch.rand(B, N, 3) * 2 - 1,
            "sizes": torch.rand(B, N, 3) * 2 - 1,
            "angles": torch.rand(B, N, 2) * 2 - 1,
            "class_labels": torch.rand(B, N, CLASS_DIM) * 2 - 1,
        }

    def test_old_path_includes_validity(self):
        model = _model(split_heads=False, loss_validity=True)
        out = model(self._batch())
        self.assertIn("loss.size_valid", out)
        self.assertIn("loss.angle_norm", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

    def test_split_path_includes_validity(self):
        model = _model(split_heads=True, loss_validity=True)
        out = model(self._batch())
        self.assertIn("loss.size_valid", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()


class TestPostProcessSizeClamp(unittest.TestCase):
    def test_negative_size_clamped(self):
        bounds = {
            "translations": (np.zeros(3, np.float32), np.ones(3, np.float32)),
            # 让 descale 后可能为负：min=-1
            "sizes": (np.full(3, -1.0, np.float32), np.ones(3, np.float32)),
            "angles": (-np.pi, np.pi),
        }
        pred = {
            "translations": np.zeros((1, 2, 3), np.float32),
            "sizes": np.full((1, 2, 3), -1.0, np.float32),  # 归一化 -1 -> descale 到 min=-1
            "angles": np.tile([1.0, 0.0], (1, 2, 1)).astype(np.float32),
            "class_labels": np.zeros((1, 2, 13), np.float32),
        }
        out = post_process(pred, bounds)
        self.assertTrue((out["sizes"].numpy() >= 0).all())


if __name__ == "__main__":
    unittest.main()
