"""校验 dataset.encoding 的编码形状与通道顺序，以及 post_process 往返。"""
import unittest

import numpy as np

import _helpers  # noqa: F401  (插入 sys.path)
from dataset.encoding import encode_sample, post_process

N_CLASSES = 15
CLASS_DIM = 14
MAX_LEN = 8


def _bounds():
    return {
        "translations": (np.zeros(3, np.float32), np.ones(3, np.float32)),
        "sizes": (np.zeros(3, np.float32), np.ones(3, np.float32)),
        "angles": (-np.pi, np.pi),
    }


def _raw_sample(L=3):
    # class one-hot: 列 0..12 材质, 13 start, 14 end；真实构件 start/end 为 0
    class_labels = np.zeros((L, N_CLASSES), np.float32)
    for i in range(L):
        class_labels[i, i % (N_CLASSES - 2)] = 1.0
    return {
        "class_labels": class_labels,
        "translations": np.full((L, 3), 0.5, np.float32),
        "sizes": np.full((L, 3), 0.5, np.float32),
        "angles": np.array([0.0, np.pi / 2, np.pi], np.float32)[:L],
    }


class TestEncoding(unittest.TestCase):
    def test_shapes(self):
        enc = encode_sample(_raw_sample(3), _bounds(), MAX_LEN,
                            "cached_diffusion_cosin_angle_wocm_prmAll", pad_zero=True)
        self.assertEqual(enc["translations"].shape, (MAX_LEN, 3))
        self.assertEqual(enc["sizes"].shape, (MAX_LEN, 3))
        self.assertEqual(enc["angles"].shape, (MAX_LEN, 2))  # (cos, sin)
        self.assertEqual(enc["class_labels"].shape, (MAX_LEN, CLASS_DIM))

    def test_channel_order_and_end_label(self):
        L = 3
        enc = encode_sample(_raw_sample(L), _bounds(), MAX_LEN,
                            "cached_diffusion_cosin_angle_wocm_prmAll", pad_zero=True)
        cls = enc["class_labels"]
        # end-label 在最后一列：真实构件 = -1，padding 槽 = +1
        np.testing.assert_allclose(cls[:L, -1], -1.0)
        np.testing.assert_allclose(cls[L:, -1], 1.0)
        # padding 槽材质列应全为 -1（仅 end 列为 +1 的 one-hot 经 *2-1）
        np.testing.assert_allclose(cls[L:, :-1], -1.0)

    def test_trans_scale_and_angle_cossin(self):
        enc = encode_sample(_raw_sample(3), _bounds(), MAX_LEN,
                            "cached_diffusion_cosin_angle_wocm_prmAll", pad_zero=True)
        # 0.5 在 [0,1] 上归一化到 [-1,1] -> 0
        np.testing.assert_allclose(enc["translations"][:3], 0.0, atol=1e-6)
        # angle 0 -> (cos, sin) = (1, 0)
        np.testing.assert_allclose(enc["angles"][0], [1.0, 0.0], atol=1e-6)

    def test_too_long_raises(self):
        with self.assertRaises(ValueError):
            encode_sample(_raw_sample(3), _bounds(), 2,
                         "cached_diffusion_cosin_angle_wocm_prmAll")

    def test_post_process_roundtrip(self):
        # 用未归一化的 translations 0.5 编码后再 post_process 应还原回 ~0.5
        enc = encode_sample(_raw_sample(3), _bounds(), MAX_LEN,
                            "cached_diffusion_cosin_angle_wocm_prmAll", pad_zero=True)
        pred = {
            "translations": enc["translations"][None, :3],
            "sizes": enc["sizes"][None, :3],
            "angles": enc["angles"][None, :3],
            "class_labels": enc["class_labels"][None, :3],
        }
        out = post_process(pred, _bounds())
        self.assertEqual(set(out.keys()),
                         {"translations", "sizes", "angles", "class_labels"})
        self.assertEqual(out["angles"].shape[-1], 1)  # (cos,sin) -> theta
        np.testing.assert_allclose(out["translations"].numpy(), 0.5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
