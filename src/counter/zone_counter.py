"""Đếm vật theo vùng: ZoneCounter (hình chữ nhật) và LaneZoneCounter (polygon)."""
import cv2
import numpy as np

from src.counter.base import BaseCounter, bbox_center

DEFAULT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


class ZoneCounter(BaseCounter):
    """Số vật ĐANG ở trong vùng hình chữ nhật tại frame hiện tại (không cộng dồn)."""

    def __init__(self, top_left, bottom_right, color=None):
        super().__init__(color)
        self.top_left = tuple(top_left)
        self.bottom_right = tuple(bottom_right)
        self.inside_ids = set()

    def _inside_zone(self, x, y):
        return (
            self.top_left[0] <= x <= self.bottom_right[0]
            and self.top_left[1] <= y <= self.bottom_right[1]
        )

    def update(self, detections):
        self.centers = []
        self.inside_ids = set()
        for detection in detections:
            center = bbox_center(detection["bbox"])
            self.centers.append(center)
            if self._inside_zone(*center):
                self.inside_ids.add(detection["track_id"])
        self.count = len(self.inside_ids)

    def _label(self):
        return f"In zone: {self.count}"

    def _draw_region(self, frame):
        cv2.rectangle(frame, self.top_left, self.bottom_right, self.color, 2)
        return frame


class LaneZoneCounter(BaseCounter):
    """Tổng số vật đã đi vào vùng polygon (một hoặc nhiều polygon, tính chung).

    min_frames: track phải có tâm trong vùng đủ số frame (cộng dồn, không cần liên tiếp) mới được
    tính. Track rác và mảnh track sinh ra khi tracker đổi ID thường rất ngắn nên bị loại, nhờ đó
    giảm đếm thừa. Đổi ID ở track dài vẫn có thể đếm thừa: đây là giới hạn của cách đếm theo vùng.
    """

    def __init__(self, points, frame_shape, colors=None, min_frames=1):
        is_multi_zone = len(points) > 0 and isinstance(points[0][0], (list, tuple))
        polygons = points if is_multi_zone else [points]
        self.zones = [np.array(zone, dtype=np.int32) for zone in polygons]

        if not colors:
            self.colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(self.zones))]
        elif isinstance(colors[0], (list, tuple)):
            self.colors = [tuple(color) for color in colors]
        else:
            self.colors = [tuple(colors)] * len(self.zones)

        super().__init__(self.colors[0])
        self.min_frames = min_frames
        self.inside_ids = set()
        self.counted_ids = set()
        self._frames_inside = {}

        # Mask cùng kích thước frame, điểm trắng (255) là thuộc vùng.
        height, width = frame_shape[:2]
        self.mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(self.mask, self.zones, 255)

    def _inside_zone(self, x, y):
        height, width = self.mask.shape
        # Kiểm tra cả số âm: chỉ số âm trong numpy sẽ lấy từ cuối mảng và cho kết quả sai.
        return 0 <= x < width and 0 <= y < height and self.mask[y, x] == 255

    def update(self, detections):
        self.centers = []
        self.inside_ids = set()

        for detection in detections:
            track_id = detection["track_id"]
            center = bbox_center(detection["bbox"])
            self.centers.append(center)

            if not self._inside_zone(*center):
                continue
            self.inside_ids.add(track_id)
            self._frames_inside[track_id] = self._frames_inside.get(track_id, 0) + 1
            if self._frames_inside[track_id] >= self.min_frames:
                self.counted_ids.add(track_id)

        self.count = len(self.counted_ids)

    def _label(self):
        return f"Total passed: {self.count}"

    def _draw_region(self, frame):
        overlay = frame.copy()
        for index, zone in enumerate(self.zones):
            color = self.colors[index % len(self.colors)]
            cv2.fillPoly(overlay, [zone], color)
            cv2.polylines(frame, [zone], True, color, 2)
        return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
