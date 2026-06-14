"""校验模型前向返回标量 loss，以及启动维度校验。"""
import os
import unittest

import torch

from _helpers import tiny_net_config, tiny_data_config, write_stats_file, make_tempdir, N_CTX
from model.builder import build_model, validate_config_dims

N_CLASSES = 15


def _synthetic_batch(B=2, N=N_CTX):
    return {
        "translations": torch.rand(B, N, 3) * 2 - 1,
        "sizes": torch.rand(B, N, 3) * 2 - 1,
        "angles": torch.rand(B, N, 2) * 2 - 1,
        "class_labels": torch.rand(B, N, 14) * 2 - 1,
    }


class TestModelForward(unittest.TestCase):
    def test_forward_returns_scalar_loss(self):
        config = {"network": tiny_net_config(loss_iou=False), "data": tiny_data_config()}
        model = build_model(config, N_CLASSES, device="cpu")
        out = model(_synthetic_batch())
        loss = out["loss"]
        self.assertEqual(loss.dim(), 0)              # 标量
        self.assertTrue(torch.isfinite(loss))
        loss.backward()                              # 可反向传播
        for key in ("loss.bbox", "loss.class", "loss.liou"):
            self.assertIn(key, out)

    def test_forward_with_iou_loss(self):
        stats = os.path.join(make_tempdir(), "dataset_stats.txt")
        write_stats_file(stats)
        net = tiny_net_config(loss_iou=True)
        net["diffusion_kwargs"]["train_stats_file"] = stats
        config = {"network": net, "data": tiny_data_config()}
        model = build_model(config, N_CLASSES, device="cpu")
        out = model(_synthetic_batch())
        self.assertTrue(torch.isfinite(out["loss"]))


class TestDimensionChecks(unittest.TestCase):
    def test_valid_passes(self):
        config = {"network": tiny_net_config(), "data": {"max_length": 128}}
        validate_config_dims(config, N_CLASSES)  # 不应抛错

    def test_point_dim_mismatch(self):
        config = {"network": tiny_net_config(), "data": {"max_length": 128}}
        config["network"]["point_dim"] = 21
        with self.assertRaises(ValueError):
            validate_config_dims(config, N_CLASSES)

    def test_class_dim_mismatch(self):
        config = {"network": tiny_net_config(), "data": {"max_length": 128}}
        with self.assertRaises(ValueError):
            validate_config_dims(config, n_classes=20)

    def test_sample_points_mismatch(self):
        config = {"network": tiny_net_config(), "data": {"max_length": 64}}
        with self.assertRaises(ValueError):
            validate_config_dims(config, N_CLASSES)


if __name__ == "__main__":
    unittest.main()
