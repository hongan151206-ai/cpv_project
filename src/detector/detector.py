"""Phát hiện và theo dõi vật bằng YOLO + ByteTrack."""
from ultralytics import YOLO


class YOLODetector:
    def __init__(
        self,
        weight_path: str,
        conf: float = 0.1,
        iou: float = 0.7,
        classes: list | None = None,
        max_det: int = 100,
        device: str | None = None,
        imgsz: int | None = None,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(weight_path)
        self.tracker = tracker
        self.predict_args = {
            "conf": conf,
            "iou": iou,
            "classes": classes,
            "max_det": max_det,
            "device": device,
        }
        if imgsz is not None:
            self.predict_args["imgsz"] = imgsz

    def detect(self, frame) -> list[dict]:
        """Trả về các vật đang được theo dõi: bbox (x1, y1, x2, y2), conf, class, track_id."""
        boxes = self.model.track(
            frame,
            persist=True,  # giữ trạng thái tracker giữa các frame, để một vật giữ nguyên track_id
            tracker=self.tracker,
            verbose=False,
            **self.predict_args,
        )[0].boxes

        # boxes.id là None khi chưa có track nào được xác nhận.
        if boxes.id is None:
            return []

        return [
            {"bbox": tuple(bbox), "conf": conf, "class": int(cls), "track_id": int(track_id)}
            for bbox, conf, cls, track_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), boxes.id.tolist(), strict=True
            )
        ]
