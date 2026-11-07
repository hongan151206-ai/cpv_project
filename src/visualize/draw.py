"""Vẽ bounding box và nhãn lên frame."""
import cv2

BOX_COLOR = (0, 0, 255)  # BGR: đỏ


def draw_boxes(frame, detections, class_names, labels=None, show_ids=True):
    """Vẽ box và nhãn dạng "Snack #12 0.87". labels cho phép đặt tên ngắn: {class_id: "tên"}."""
    labels = labels or {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, frame.shape[0] / 1400)
    thickness = max(1, round(scale * 1.5))

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        cls_id = detection["class"]

        text = labels.get(cls_id, class_names[cls_id])
        if show_ids:
            text += f" #{detection['track_id']}"
        text += f" {detection['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness + 1)

        # Nền của nhãn; đẩy xuống trong box nếu vật sát mép trên để chữ không bị cắt.
        (width, height), _ = cv2.getTextSize(text, font, scale, thickness)
        top = max(0, y1 - height - 6)
        cv2.rectangle(frame, (x1, top), (x1 + width, top + height + 6), BOX_COLOR, -1)
        cv2.putText(frame, text, (x1, top + height + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame
