"""校验 config.loader 的相对路径解析（相对 YAML 文件，而非 shell cwd）。"""
import os
import unittest

from _helpers import make_tempdir
from config.loader import load_config, save_config


class TestConfigPathResolution(unittest.TestCase):
    def test_relative_paths_resolved_against_yaml_dir(self):
        tmp = make_tempdir()
        cfg_dir = os.path.join(tmp, "config")
        os.makedirs(cfg_dir, exist_ok=True)
        cfg_path = os.path.join(cfg_dir, "x.yaml")

        save_config(
            {
                "data": {
                    "dataset_directory": "../data/Boxes",
                    "annotation_file": "splits.csv",
                },
                "network": {
                    "diffusion_kwargs": {"train_stats_file": "../stats.txt"},
                },
            },
            cfg_path,
        )

        loaded = load_config(cfg_path)
        data = loaded["data"]

        # 相对路径应解析为相对 yaml 所在目录的绝对路径
        self.assertEqual(
            data["dataset_directory"],
            os.path.normpath(os.path.join(tmp, "data", "Boxes")),
        )
        self.assertEqual(
            data["annotation_file"],
            os.path.normpath(os.path.join(cfg_dir, "splits.csv")),
        )
        self.assertEqual(
            loaded["network"]["diffusion_kwargs"]["train_stats_file"],
            os.path.normpath(os.path.join(tmp, "stats.txt")),
        )
        self.assertTrue(os.path.isabs(data["dataset_directory"]))

    def test_missing_config_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config(os.path.join(make_tempdir(), "nope.yaml"))


if __name__ == "__main__":
    unittest.main()
