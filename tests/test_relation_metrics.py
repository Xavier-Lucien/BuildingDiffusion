"""校验关系诊断指标的核心逻辑。"""
import unittest

import numpy as np

import _helpers  # noqa: F401  (插入 sys.path)
from utils.relation_metrics import (
    diagnose_scene,
    aggregate_scene_metrics,
    scene_from_post_process,
)


class TestDiagnoseScene(unittest.TestCase):
    def test_window_attach_and_floating(self):
        # 一面墙（y 方向很薄）；一个窗贴墙，一个窗飘在远处
        centers = np.array([
            [0.0, 0.0, 0.5],     # wall
            [0.0, 0.05, 0.5],    # window 贴墙（与墙在 y 上接触）
            [5.0, 5.0, 5.0],     # window 飘走
        ], np.float32)
        sizes = np.array([
            [1.0, 0.1, 1.0],
            [0.2, 0.05, 0.2],
            [0.2, 0.05, 0.2],
        ], np.float32)
        labels = ["wall", "window", "window"]

        m = diagnose_scene(centers, sizes, labels)
        self.assertEqual(m["n_window"], 2)
        self.assertEqual(m["window_attached"], 1)
        self.assertEqual(m["floating_window"], 1)
        self.assertEqual(m["n_wall"], 1)

    def test_invalid_size(self):
        centers = np.zeros((2, 3), np.float32)
        sizes = np.array([[1.0, 1.0, 1.0], [0.5, -0.1, 0.5]], np.float32)
        m = diagnose_scene(centers, sizes, ["wall", "window"])
        self.assertEqual(m["invalid_size"], 1)

    def test_empty_scene(self):
        m = diagnose_scene(np.zeros((0, 3)), np.zeros((0, 3)), [])
        self.assertEqual(m["n_objects"], 0)
        self.assertIsNone(m["roof_wall_footprint_iou"])

    def test_roof_wall_footprint(self):
        # wall 与 roof 的 x-y footprint 完全重合 -> IoU≈1，水平偏移≈0
        centers = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.2]], np.float32)
        sizes = np.array([[2.0, 2.0, 1.0], [2.0, 2.0, 0.2]], np.float32)
        m = diagnose_scene(centers, sizes, ["wall", "roof"])
        self.assertAlmostEqual(m["roof_wall_footprint_iou"], 1.0, places=5)
        self.assertLess(m["roof_wall_alignment_error"], 1e-5)

    def test_no_wall_means_no_attach(self):
        m = diagnose_scene(
            np.array([[0.0, 0.0, 0.0]], np.float32),
            np.array([[0.2, 0.2, 0.2]], np.float32),
            ["window"],
        )
        self.assertEqual(m["n_window"], 1)
        self.assertEqual(m["window_attached"], 0)
        self.assertEqual(m["floating_window"], 1)


class TestAggregate(unittest.TestCase):
    def test_empty_generation_rate(self):
        scenes = [
            diagnose_scene(np.zeros((0, 3)), np.zeros((0, 3)), []),
            diagnose_scene(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0]], np.float32),
                np.array([[1.0, 0.1, 1.0], [0.2, 0.05, 0.2]], np.float32),
                ["wall", "window"],
            ),
        ]
        agg = aggregate_scene_metrics(scenes)
        self.assertEqual(agg["n_scenes"], 2)
        self.assertAlmostEqual(agg["empty_generation_rate"], 0.5)
        self.assertEqual(agg["window_attach_rate"], 1.0)

    def test_empty_list(self):
        self.assertEqual(aggregate_scene_metrics([])["n_scenes"], 0)


class TestSceneFromPostProcess(unittest.TestCase):
    def test_argmax_labels_and_size_half(self):
        class_names = ["wall", "window", "door"]
        processed = {
            "translations": np.zeros((1, 2, 3), np.float32),
            "sizes": np.ones((1, 2, 3), np.float32),
            "angles": np.zeros((1, 2, 1), np.float32),
            "class_labels": np.array([[[5.0, 0.0, 0.0], [0.0, 9.0, 0.0]]], np.float32),
        }
        centers, sizes_full, labels = scene_from_post_process(
            processed, class_names, size_half=True
        )
        self.assertEqual(labels, ["wall", "window"])
        np.testing.assert_allclose(sizes_full, 2.0)  # size_half -> 全尺寸翻倍
        self.assertEqual(centers.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
