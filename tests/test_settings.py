"""Kiểm tra nạp cấu hình và logging. Chạy: py -m unittest discover -s tests -t ."""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from core.logging_setup import setup_logging
from core.settings_loader import PROJECT_ROOT, ConfigError, load_settings

BROKEN_LINE = {"counters": [{"type": "line", "start": [1, 1], "end": [1, 1]}]}


def env_without_overrides(**extra):
    """Môi trường sạch để test không phụ thuộc biến của máy; chặn cả việc đọc file .env."""
    clean = {key: value for key, value in os.environ.items() if not key.startswith(("INPUT_", "DETECTOR_", "DATA_", "SAVER_", "TRACKER_", "APP_"))}
    clean.update(extra)
    return mock.patch.dict(os.environ, clean, clear=True)


class SettingsTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("core.settings_loader.load_dotenv")
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_preset(self, content):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "preset.yaml"
        base = yaml.safe_load((PROJECT_ROOT / "config" / "snack.yaml").read_text(encoding="utf-8"))
        base.update(content)
        path.write_text(yaml.safe_dump(base), encoding="utf-8")
        return path

    def test_presets_are_valid_and_paths_absolute(self):
        with env_without_overrides():
            for name, model in (("snack", "snack"), ("coconut", "coconut")):
                settings = load_settings(f"config/{name}.yaml")
                self.assertEqual(settings["detector"]["type"], model)
                self.assertTrue(Path(settings["input"]["path"]).is_absolute())
                self.assertTrue(Path(settings["tracker"]["config"]).is_file())

    def test_default_preset_is_snack(self):
        with env_without_overrides():
            self.assertEqual(load_settings()["detector"]["type"], "snack")

    def test_independent_of_working_directory(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory, env_without_overrides():
            os.chdir(directory)
            try:
                self.assertEqual(load_settings("config/coconut.yaml")["detector"]["type"], "coconut")
            finally:
                os.chdir(previous)

    def test_priority_yaml_then_env_then_cli(self):
        with env_without_overrides():
            self.assertEqual(load_settings()["detector"]["conf_threshold"], 0.1)
        with env_without_overrides(DETECTOR_CONF_THRESHOLD="0.3", INPUT_DISPLAY="false", DETECTOR_CLASSES="0,2"):
            settings = load_settings()
            self.assertEqual(settings["detector"]["conf_threshold"], 0.3)
            self.assertIs(settings["input"]["display"], False)
            self.assertEqual(settings["detector"]["classes"], [0, 2])
            settings = load_settings(overrides={("detector", "conf_threshold"): 0.4})
            self.assertEqual(settings["detector"]["conf_threshold"], 0.4)

    def test_invalid_env_value_gives_config_error(self):
        with env_without_overrides(INPUT_VID_STRIDE="abc"), self.assertRaises(ConfigError):
            load_settings()

    def test_missing_files_are_all_reported(self):
        with env_without_overrides(), self.assertRaises(ConfigError) as caught:
            load_settings(overrides={("detector", "type"): "khong-co", ("input", "path"): "khong-co.mp4"})
        message = str(caught.exception)
        self.assertIn("khong-co.pt", message)
        self.assertIn("khong-co.mp4", message)

    def test_invalid_counters_are_rejected(self):
        cases = [
            BROKEN_LINE,
            {"counters": []},
            {"counters": [{"type": "circle"}]},
            {"counters": [{"type": "line", "start": [0, 0], "end": [1, 1], "direction": "up"}]},
            {"counters": [{"type": "lane", "points": [[0, 0], [1, 1]]}]},
            {"counters": [{"type": "lane", "points": [[0, 0], [1, 1], [2, 0]], "min_frames": 0}]},
            {"counters": [{"type": "zone", "top_left": [5, 5], "bottom_right": [1, 1]}]},
        ]
        for case in cases:
            with self.subTest(case=case), env_without_overrides(), self.assertRaises(ConfigError):
                load_settings(self.write_preset(case))

    def test_missing_preset_gives_config_error(self):
        with env_without_overrides(), self.assertRaises(ConfigError):
            load_settings("config/khong-co.yaml")


class CommandLineTest(unittest.TestCase):
    def test_arguments_map_to_overrides(self):
        import argparse

        import run

        args = argparse.Namespace(input=None, model="coconut", conf=0.3, stride=2, no_display=True, save_video=True)
        self.assertEqual(
            run.build_overrides(args),
            {
                ("detector", "type"): "coconut",
                ("detector", "conf_threshold"): 0.3,
                ("input", "vid_stride"): 2,
                ("input", "display"): False,
                ("saver", "save_videos"): True,
            },
        )

    def test_unset_arguments_do_not_override_config(self):
        import argparse

        import run

        args = argparse.Namespace(input=None, model=None, conf=None, stride=None, no_display=False, save_video=False)
        self.assertEqual(run.build_overrides(args), {})

    def test_conf_zero_is_a_valid_override(self):
        import argparse

        import run

        args = argparse.Namespace(input=None, model=None, conf=0.0, stride=None, no_display=False, save_video=False)
        self.assertEqual(run.build_overrides(args), {("detector", "conf_threshold"): 0.0})


class LoggingTest(unittest.TestCase):
    def test_loggers_created_before_setup_stay_enabled(self):
        # Hồi quy: dictConfig mặc định tắt mọi logger đã tạo, khiến không có dòng log nào.
        logger = logging.getLogger("tests.created_before_setup")
        setup_logging()
        self.addCleanup(self._close_handlers)
        self.assertFalse(logger.disabled)
        self.assertTrue(logger.isEnabledFor(logging.INFO))

    @staticmethod
    def _close_handlers():
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
