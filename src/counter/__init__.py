from .line_counter import LineCounter
from .zone_counter import LaneZoneCounter, ZoneCounter


def build_counter(config, frame_shape):
    """Tạo counter từ một phần tử trong mục `counters` của config."""
    counter_type = config["type"]

    if counter_type == "line":
        return LineCounter(
            tuple(config["start"]),
            tuple(config["end"]),
            color=config.get("color"),
            direction=config.get("direction", "both"),
            max_missing_frames=config.get("max_missing_frames", 30),
            duplicate_iou=config.get("duplicate_iou"),
            duplicate_frames=config.get("duplicate_frames", 0),
        )
    if counter_type == "zone":
        return ZoneCounter(tuple(config["top_left"]), tuple(config["bottom_right"]), color=config.get("color"))
    if counter_type == "lane":
        return LaneZoneCounter(
            config["points"],
            frame_shape,
            colors=config.get("colors", config.get("color")),
            min_frames=config.get("min_frames", 1),
        )
    raise ValueError(f"Unknown counter type: {counter_type}")


__all__ = ["LineCounter", "ZoneCounter", "LaneZoneCounter", "build_counter"]
