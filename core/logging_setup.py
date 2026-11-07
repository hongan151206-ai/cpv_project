"""Thiết lập logging từ config/logging.yaml; file log luôn nằm trong <project>/logs."""
import logging.config

import yaml

from core.settings_loader import PROJECT_ROOT


def setup_logging():
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    with open(PROJECT_ROOT / "config" / "logging.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["handlers"]["file"]["filename"] = str(logs_dir / "application.log")
    logging.config.dictConfig(config)
