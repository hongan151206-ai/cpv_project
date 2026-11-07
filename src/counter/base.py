"""Phần dùng chung của các counter: tính tâm bbox và vẽ nhãn số đếm."""
import cv2

CENTER_COLOR = (0, 255, 255)
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_count_label(frame, text, color, slot=0):
    """Vẽ nhãn ở góc trên-trái, cỡ chữ theo chiều cao frame để đọc được ở mọi độ phân giải.

    slot: thứ tự dòng, để nhiều counter không vẽ đè lên nhau.
    """
    scale = frame.shape[0] / 720
    thickness = max(1, round(2 * scale))
    (width, height), baseline = cv2.getTextSize(text, LABEL_FONT, scale, thickness)
    margin = max(4, round(12 * scale))
    top = margin + slot * (height + baseline + 2 * margin)
    bottom = top + height + baseline + margin
    cv2.rectangle(frame, (margin, top), (2 * margin + width, bottom), (30, 30, 30), -1)
    cv2.putText(
        frame,
        text,
        (margin + margin // 2, top + height + margin // 2),
        LABEL_FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


class BaseCounter:
    """Giao diện chung: update(detections) rồi draw(frame). Lớp con cài update và _draw_region."""

    def __init__(self, color=None):
        self.color = tuple(color) if color else (0, 255, 0)
        self.count = 0
        self.centers = []

    def update(self, detections):
        raise NotImplementedError

    def _label(self):
        return f"Count: {self.count}"

    def _draw_region(self, frame):
        return frame

    def draw(self, frame, show_count=True, show_centers=True, slot=0):
        frame = self._draw_region(frame)
        if show_centers:
            for center in self.centers:
                cv2.circle(frame, center, max(3, frame.shape[0] // 200), CENTER_COLOR, -1)
        if show_count:
            frame = draw_count_label(frame, self._label(), self.color, slot)
        return frame
