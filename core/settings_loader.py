"""Nạp cấu hình: config/settings.yaml (chung) + preset + biến môi trường + tham số dòng lệnh.

Thứ tự ưu tiên tăng dần: settings.yaml < preset (--config) < .env / biến môi trường < tham số CLI.
Đường dẫn tương đối trong config luôn được tính từ thư mục project, nên có thể chạy từ bất kỳ đâu.
"""
import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_PRESET = PROJECT_ROOT / "config" / "snack.yaml"

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
LINE_DIRECTIONS = {"both", "positive", "negative"}


class ConfigError(Exception):
    """Cấu hình không hợp lệ; thông báo đã viết sẵn để hiển thị cho người dùng."""


def _to_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_optional_str(value):
    return None if value.strip().lower() in {"", "null", "none"} else value


def _to_classes(value):
    if value.strip().lower() in {"", "null", "none"}:
        return None
    return [int(item) for item in value.split(",")]


# Biến môi trường -> (vị trí trong config, hàm chuyển đổi)
ENV_OVERRIDES = {
    "APP_NAME": (("app", "name"), str),
    "DATA_OUTPUT": (("data", "output"), str),
    "DATA_MODEL": (("data", "model"), str),
    "INPUT_PATH": (("input", "path"), str),
    "INPUT_VID_STRIDE": (("input", "vid_stride"), int),
    "INPUT_DISPLAY": (("input", "display"), _to_bool),
    "INPUT_REALTIME": (("input", "realtime"), _to_bool),
    "DETECTOR_TYPE": (("detector", "type"), str),
    "DETECTOR_CONF_THRESHOLD": (("detector", "conf_threshold"), float),
    "DETECTOR_IOU_THRESHOLD": (("detector", "iou_threshold"), float),
    "DETECTOR_MAX_DET": (("detector", "max_det"), int),
    "DETECTOR_IMGSZ": (("detector", "imgsz"), int),
    "DETECTOR_DEVICE": (("detector", "device"), _to_optional_str),
    "DETECTOR_CLASSES": (("detector", "classes"), _to_classes),
    "SAVER_SAVE_IMAGES": (("saver", "save_images"), _to_bool),
    "SAVER_SAVE_VIDEOS": (("saver", "save_videos"), _to_bool),
    "SAVER_SAVE_DETECTIONS": (("saver", "save_detections"), _to_bool),
    "SAVER_SAVE_CROPS": (("saver", "save_crops"), _to_bool),
    "TRACKER_CONFIG": (("tracker", "config"), str),
}


def _read_yaml(path):
    try:
        with open(path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        raise ConfigError(f"Không tìm thấy file cấu hình: {path}") from None
    except yaml.YAMLError as error:
        raise ConfigError(f"File cấu hình {path} sai cú pháp YAML:\n{error}") from None


def _deep_merge(base, override):
    """Gộp override vào base; dict được gộp đệ quy, các kiểu khác (kể cả list) bị thay thế."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_path(settings, path, value):
    node = settings
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _apply_env_overrides(settings):
    for name, (path, convert) in ENV_OVERRIDES.items():
        raw = os.getenv(name)
        if raw is None or raw == "":
            continue
        try:
            _set_path(settings, path, convert(raw))
        except ValueError:
            raise ConfigError(f"Biến môi trường {name}={raw!r} không hợp lệ") from None


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resolve_paths(settings):
    settings["input"]["path"] = str(_absolute(settings["input"]["path"]))
    settings["data"]["model"] = str(_absolute(settings["data"]["model"]))
    settings["data"]["output"] = str(_absolute(settings["data"]["output"]))
    settings["tracker"]["config"] = str(_absolute(settings["tracker"]["config"]))


def model_path(settings):
    return Path(settings["data"]["model"]) / f"{settings['detector']['type']}.pt"


def _is_point(value):
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _validate_counter(index, counter):
    prefix = f"counters[{index}]"
    counter_type = counter.get("type") if isinstance(counter, dict) else None

    if counter_type == "line":
        start, end = counter.get("start"), counter.get("end")
        if not (_is_point(start) and _is_point(end)):
            return [f"{prefix}: line cần 'start' và 'end' dạng [x, y]"]
        errors = []
        if list(start) == list(end):
            errors.append(f"{prefix}: 'start' và 'end' của line phải khác nhau")
        direction = counter.get("direction", "both")
        if direction not in LINE_DIRECTIONS:
            errors.append(f"{prefix}: 'direction' phải thuộc {sorted(LINE_DIRECTIONS)}")
        return errors

    if counter_type == "zone":
        top_left, bottom_right = counter.get("top_left"), counter.get("bottom_right")
        if not (_is_point(top_left) and _is_point(bottom_right)):
            return [f"{prefix}: zone cần 'top_left' và 'bottom_right' dạng [x, y]"]
        if not (top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]):
            return [f"{prefix}: 'top_left' phải nằm trên-trái 'bottom_right'"]
        return []

    if counter_type == "lane":
        points = counter.get("points")
        if not isinstance(points, list) or not points:
            return [f"{prefix}: lane cần 'points' (một hoặc nhiều polygon)"]
        polygons = [points] if _is_point(points[0]) else points
        errors = []
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3 or not all(_is_point(p) for p in polygon):
                errors.append(f"{prefix}: mỗi polygon cần ít nhất 3 điểm dạng [x, y]")
        min_frames = counter.get("min_frames", 1)
        if not isinstance(min_frames, int) or min_frames < 1:
            errors.append(f"{prefix}: 'min_frames' phải là số nguyên >= 1")
        return errors

    return [f"{prefix}: 'type' phải là line, zone hoặc lane (đang là {counter_type!r})"]


def validate_settings(settings):
    """Kiểm tra cấu hình và gom mọi lỗi vào một ConfigError, tránh KeyError khó hiểu lúc đang chạy."""
    errors = []

    video = Path(settings["input"]["path"])
    if not video.is_file():
        errors.append(f"Không tìm thấy video: {video}")
    elif video.suffix.lower() not in VIDEO_SUFFIXES:
        errors.append(f"Định dạng video không hỗ trợ: {video.suffix} (hỗ trợ {sorted(VIDEO_SUFFIXES)})")

    stride = settings["input"].get("vid_stride", 1)
    if not isinstance(stride, int) or stride < 1:
        errors.append("input.vid_stride phải là số nguyên >= 1")

    weights = model_path(settings)
    if not weights.is_file():
        errors.append(f"Không tìm thấy model: {weights} (kiểm tra detector.type)")

    for key in ("conf_threshold", "iou_threshold"):
        value = settings["detector"].get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            errors.append(f"detector.{key} phải nằm trong khoảng [0, 1]")

    tracker = Path(settings["tracker"]["config"])
    if not tracker.is_file():
        errors.append(f"Không tìm thấy cấu hình tracker: {tracker}")

    counters = settings.get("counters")
    if not isinstance(counters, list) or not counters:
        errors.append("Cần khai báo ít nhất một counter trong 'counters'")
    else:
        for index, counter in enumerate(counters):
            errors.extend(_validate_counter(index, counter))

    if errors:
        raise ConfigError("Cấu hình không hợp lệ:\n" + "\n".join(f"  - {error}" for error in errors))


def load_settings(config_path=None, overrides=None):
    """Trả về config đã gộp, đã chuẩn hóa đường dẫn và đã kiểm tra.

    config_path: file preset (mặc định config/snack.yaml).
    overrides: {("detector", "type"): "coconut", ...}, thường đến từ tham số dòng lệnh.
    """
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    preset = _absolute(config_path) if config_path else DEFAULT_PRESET
    settings = _deep_merge(_read_yaml(BASE_CONFIG), _read_yaml(preset))
    _apply_env_overrides(settings)
    for path, value in (overrides or {}).items():
        _set_path(settings, path, value)

    if not settings.get("input", {}).get("path") or not settings.get("detector", {}).get("type"):
        raise ConfigError(f"Preset {preset} phải khai báo input.path và detector.type")

    _resolve_paths(settings)
    validate_settings(settings)
    return settings
