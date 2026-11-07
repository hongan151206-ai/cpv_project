"""Đếm vật khi tâm vật vượt qua một đoạn thẳng, theo mọi góc và có thể lọc theo hướng."""
from dataclasses import dataclass

import cv2

from src.counter.base import BaseCounter, bbox_center


def _iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


@dataclass
class _TrackState:
    point: tuple  # vị trí gần nhất không nằm đúng trên đường thẳng
    side: int  # +1 hoặc -1: phía của đường thẳng tại `point`
    last_seen: int


class LineCounter(BaseCounter):
    """Mỗi track_id được đếm tối đa một lần.

    direction:
        "both"     - đếm cả hai chiều.
        "positive" - chỉ đếm khi đi từ phía âm sang phía dương.
        "negative" - ngược lại.
    Quy ước phía: với đường nằm ngang vẽ từ trái sang phải, phía dương ở bên dưới, nên "positive"
    nghĩa là vật đi từ trên xuống. Đảo start và end sẽ đảo phía.

    max_missing_frames: giữ trạng thái của track bị mất detection trong bao nhiêu frame, để vật
    chỉ mất dấu thoáng qua ngay lúc vượt đường vẫn được đếm.

    duplicate_iou, duplicate_frames: YOLO26 không dùng NMS nên đôi khi trả hai box chồng lên cùng một
    vật, tracker gán hai ID khác nhau. Khi duplicate_iou được đặt, một track vượt đường mà box trùng
    (IoU >= duplicate_iou) với box vừa được đếm trong duplicate_frames frame gần nhất (0 = cùng frame)
    bị coi là bản sao và không đếm thêm. Mặc định tắt: vật thật có thể chồng lên nhau (gói snack),
    khi đó bật lên sẽ đếm thiếu.
    """

    def __init__(
        self, start, end, color=None, direction="both", max_missing_frames=30, duplicate_iou=None, duplicate_frames=0
    ):
        super().__init__(color)
        if tuple(start) == tuple(end):
            raise ValueError("start and end of a line must differ")
        if direction not in ("both", "positive", "negative"):
            raise ValueError(f"Unknown line direction: {direction}")
        self.start = tuple(start)
        self.end = tuple(end)
        self.direction = direction
        self.max_missing_frames = max_missing_frames
        self.duplicate_iou = duplicate_iou
        self.duplicate_frames = duplicate_frames
        self.duplicates_skipped = 0
        self._recent_counted = []  # (frame, bbox) của các lần đếm gần đây
        self.counted_ids = set()
        self.events = []  # mỗi lần đếm: frame, track_id, tâm tại thời điểm vượt đường
        self._tracks = {}
        self._frame_index = 0

    def _signed_distance(self, point):
        """Tích có hướng: dấu cho biết điểm nằm ở phía nào của đường thẳng start -> end."""
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        return dx * (point[1] - self.start[1]) - dy * (point[0] - self.start[0])

    def _crosses_segment(self, previous, current):
        """Giao điểm của quãng di chuyển với đường thẳng có nằm trong đoạn start-end không."""
        d0, d1 = self._signed_distance(previous), self._signed_distance(current)
        ratio = d0 / (d0 - d1)
        x = previous[0] + ratio * (current[0] - previous[0])
        y = previous[1] + ratio * (current[1] - previous[1])
        dx, dy = self.end[0] - self.start[0], self.end[1] - self.start[1]
        position = ((x - self.start[0]) * dx + (y - self.start[1]) * dy) / (dx * dx + dy * dy)
        return 0 <= position <= 1

    def _direction_allowed(self, new_side):
        if self.direction == "both":
            return True
        return new_side > 0 if self.direction == "positive" else new_side < 0

    def _is_duplicate(self, bbox):
        if self.duplicate_iou is None:
            return False
        self._recent_counted = [
            item for item in self._recent_counted if self._frame_index - item[0] <= self.duplicate_frames
        ]
        return any(_iou(bbox, other) >= self.duplicate_iou for _, other in self._recent_counted)

    def update(self, detections):
        self._frame_index += 1
        self.centers = []

        for detection in detections:
            track_id = detection["track_id"]
            center = bbox_center(detection["bbox"])
            self.centers.append(center)

            distance = self._signed_distance(center)
            state = self._tracks.get(track_id)
            if state is not None:
                state.last_seen = self._frame_index

            # Tâm nằm đúng trên đường: chưa kết luận, chờ frame sau để biết đã sang phía nào.
            if distance == 0:
                continue

            side = 1 if distance > 0 else -1
            if state is None:
                self._tracks[track_id] = _TrackState(center, side, self._frame_index)
                continue

            if (
                side != state.side
                and track_id not in self.counted_ids
                and self._direction_allowed(side)
                and self._crosses_segment(state.point, center)
            ):
                self.counted_ids.add(track_id)
                if self._is_duplicate(detection["bbox"]):
                    self.duplicates_skipped += 1
                else:
                    self.count += 1
                    self._recent_counted.append((self._frame_index, detection["bbox"]))
                    self.events.append({"frame": self._frame_index, "track_id": track_id, "center": center})

            state.point = center
            state.side = side

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if self._frame_index - state.last_seen > self.max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _label(self):
        return f"Passed: {self.count}"

    def _draw_region(self, frame):
        cv2.line(frame, self.start, self.end, self.color, max(2, frame.shape[0] // 270))
        return frame
