"""Lưu kết quả của một lần chạy vào thư mục riêng: outputs/<thời gian>_<tên video>/."""
import csv
import logging
from datetime import datetime
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class ResultSaver:
    def __init__(
        self,
        save_dir="outputs",
        run_name="run",
        save_frame=False,
        save_video=False,
        save_txt=False,
        save_crop=False,
    ):
        self.save_frame_flag = save_frame
        self.save_video_flag = save_video
        self.save_txt_flag = save_txt
        self.save_crop_flag = save_crop
        self._writer = None

        # Chỉ tạo thư mục khi thật sự có thứ cần lưu.
        self.run_dir = None
        if save_frame or save_video or save_txt or save_crop:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(save_dir) / f"{timestamp}_{run_name}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Saving results to %s", self.run_dir)

    def save_raw(self, frame, detections, frame_id):
        """Lưu phần cần frame gốc (chưa vẽ): file txt và ảnh crop."""
        if self.save_txt_flag:
            self._save_txt(detections, frame_id)
        if self.save_crop_flag:
            self._save_crop(frame, detections, frame_id)

    def save_annotated(self, frame, frame_id, fps):
        """Lưu phần cần frame đã vẽ: ảnh và video kết quả."""
        if self.save_frame_flag:
            self._write_image(self.run_dir / f"frame_{frame_id:05d}.jpg", frame)
        if self.save_video_flag:
            self._write_video(frame, fps)

    def save_events(self, events):
        """Ghi từng lần đếm vào events.csv (cột: counter, n, frame, seconds, track_id, x, y)."""
        if self.run_dir is None:
            return
        path = self.run_dir / "events.csv"
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["counter", "n", "frame", "seconds", "track_id", "x", "y"])
            writer.writerows(
                [e["counter"], e["n"], e["frame"], f"{e['seconds']:.2f}", e["track_id"], e["x"], e["y"]]
                for e in events
            )
        logger.info("Saved %d counting events to %s", len(events), path)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _write_image(self, path, image):
        # cv2.imwrite không ném lỗi khi thất bại, chỉ trả về False.
        if not cv2.imwrite(str(path), image):
            logger.warning("Could not write image: %s", path)

    def _write_video(self, frame, fps):
        if self._writer is None:
            path = self.run_dir / "result.mp4"
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                logger.warning("Could not open video writer: %s", path)
                self.save_video_flag = False
                return
            self._writer = writer
        self._writer.write(frame)

    def _save_txt(self, detections, frame_id):
        path = self.run_dir / f"frame_{frame_id:05d}.txt"
        with open(path, "w", encoding="utf-8") as file:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                file.write(
                    f"{det['class']} {det['track_id']} {det['conf']:.3f} "
                    f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n"
                )

    def _save_crop(self, frame, detections, frame_id):
        height, width = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            path = self.run_dir / f"crop_{frame_id:05d}_id{det['track_id']}.jpg"
            self._write_image(path, frame[y1:y2, x1:x2])
