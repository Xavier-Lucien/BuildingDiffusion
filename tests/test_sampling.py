"""校验采样输出可以通过 post_process。"""
import unittest

import numpy as np
import torch

from _helpers import tiny_net_config, tiny_data_config, N_CTX
from model.builder import build_model
from dataset.encoding import post_process

N_CLASSES = 15


def _bounds():
    return {
        "translations": (np.zeros(3, np.float32), np.ones(3, np.float32)),
        "sizes": (np.zeros(3, np.float32), np.ones(3, np.float32)),
        "angles": (-np.pi, np.pi),
    }


class TestSampling(unittest.TestCase):
    def test_generate_then_post_process(self):
        config = {"network": tiny_net_config(loss_iou=False), "data": tiny_data_config()}
        model = build_model(config, N_CLASSES, device="cpu")
        model.eval()

        N = N_CTX
        samples = model.generate_layout(
            batch_size=1, num_points=N, point_dim=22, device="cpu", clip_denoised=True
        )
        self.assertIsInstance(samples, list)
        self.assertEqual(len(samples), 1)

        raw = samples[0]
        # 拆分出的字段维度正确（构件数 K 取决于 end-label 过滤，可能为 0）
        self.assertEqual(raw["translations"].shape[-1], 3)
        self.assertEqual(raw["sizes"].shape[-1], 3)
        self.assertEqual(raw["angles"].shape[-1], 2)
        self.assertEqual(raw["class_labels"].shape[-1], N_CLASSES - 2)

        processed = post_process(raw, _bounds())
        self.assertEqual(
            set(processed.keys()),
            {"translations", "sizes", "angles", "class_labels"},
        )
        self.assertEqual(processed["angles"].shape[-1], 1)  # (cos,sin) -> theta
        self.assertTrue(torch.isfinite(processed["translations"]).all())


if __name__ == "__main__":
    unittest.main()
