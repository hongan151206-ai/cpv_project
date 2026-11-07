"""Pipeline xử lý video: phát hiện + theo dõi -> đếm -> vẽ -> lưu/hiển thị."""
import logging
import time
from pathlib import Path

import cv2

from core.settings_loader import model_path
from src.counter import build_counter
from src.detector.detector import YOLODetector
from src.saver.saver import ResultSaver
from src.visualize.draw import draw_boxes

logger = logging.getLogger(__name__)

MAX_WINDOW_SIZE = (1280, 720)
QUIT_KEYS = {27, ord("q")}  # ESC hoặc q


class DetectionPipeline:
    def __init__(self, config):
        self.config = config
        self.window_name = config.get("app", {}).get("name", "Detection")
        self.visualizer = config.get("visualizer", {})
        self.counters = []
        self._display_size = MAX_WINDOW_SIZE

        detector = config["detector"]
        self.detector = YOLODetector(
            weight_path=str(model_path(config)),
            conf=detector["conf_threshold"],
            iou=detector["iou_threshold"],
            classes=detector.get("classes"),
            max_det=detector.get("max_det", 100),
            device=detector.get("device"),
            imgsz=detector.get("imgsz"),
            tracker=config["tracker"]["config"],
        )
        logger.info("Detection pipeline initialized (model: %s)", detector["type"])

    def _annotate(self, frame, detections):
        if self.visualizer.get("show_detections", True):
            min_conf = self.visualizer.get("min_display_conf", 0)
            frame = draw_boxes(
                frame,
                [detection for detection in detections if detection["conf"] >= min_conf],
                self.detector.model.names,
                labels=self.visualizer.get("labels"),
                show_ids=self.visualizer.get("show_tracks", True),
            )
        for slot, counter in enumerate(self.counters):
            frame = counter.draw(
                frame,
                show_count=self.visualizer.get("show_count", True),
                show_centers=self.visualizer.get("show_tracks", True),
                slot=slot,
            )
        return frame

    def _open_window(self, frame):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        height, width = frame.shape[:2]
        ratio = min(MAX_WINDOW_SIZE[0] / width, MAX_WINDOW_SIZE[1] / height, 1)
        self._display_size = (int(width * ratio), int(height * ratio))
        cv2.resizeWindow(self.window_name, *self._display_size)

    def _window_exists(self):
        """False khi cửa sổ đã bị đóng. WND_PROP_VISIBLE không dùng được: nó trả 0 cả khi cửa sổ vừa bị
        hủy lẫn lúc chỉ tạm ẩn, còn AUTOSIZE thì ném lỗi chỉ khi cửa sổ không còn tồn tại."""
        try:
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)
        except cv2.error:
            return False
        return True

    def _show(self, frame):
        """Hiển thị frame; trả về False nếu người dùng muốn dừng (ESC, q, hoặc đóng cửa sổ)."""
        # Thu nhỏ trước khi đưa vào cửa sổ nhanh hơn để GUI tự co (đo được ~12 ms -> ~5 ms/frame 1080p).
        # Chỉ ảnh hiển thị bị thu nhỏ; video và ảnh lưu vẫn giữ độ phân giải gốc.
        if frame.shape[1] > self._display_size[0]:
            frame = cv2.resize(frame, self._display_size, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            logger.info("Quit key pressed (code %d)", key)
            return False
        if not self._window_exists():
            logger.info("Display window closed")
            return False
        return True

    def _collect_events(self, source_fps):
        """Gom các lần đếm của mọi line counter, sắp theo thời gian, để ghi ra events.csv."""
        rows = []
        for index, counter in enumerate(self.counters):
            for number, event in enumerate(getattr(counter, "events", ()), start=1):
                frame = event.get("source_frame", event["frame"])
                x, y = event["center"]
                rows.append(
                    {
                        "counter": index,
                        "n": number,
                        "frame": frame,
                        "seconds": frame / source_fps,
                        "track_id": event["track_id"],
                        "x": int(x),
                        "y": int(y),
                    }
                )
        return sorted(rows, key=lambda row: (row["frame"], row["counter"]))

    def run_video(self, video_path, vid_stride=1, display=True, realtime=True):
        """Xử lý cả video và trả về kết quả dạng dict (số đếm, số frame, FPS).

        vid_stride > 1: chỉ xử lý mỗi vid_stride frame, các frame giữa bị bỏ hẳn để counter không
        nhận lại detection cũ. Nên giữ bằng 1 khi cần đếm chính xác vì tracker sẽ dễ mất dấu hơn.
        realtime: khi hiển thị, giữ đúng tốc độ video gốc thay vì phát nhanh nhất có thể.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Could not open video: {video_path}")

        saver_config = self.config.get("saver", {})
        saver = ResultSaver(
            save_dir=self.config["data"]["output"],
            run_name=Path(video_path).stem,
            save_frame=saver_config.get("save_images", False),
            save_video=saver_config.get("save_videos", False),
            save_txt=saver_config.get("save_detections", False),
            save_crop=saver_config.get("save_crops", False),
        )

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        output_fps = source_fps / vid_stride
        frame_interval = 1 / output_fps

        frame_id = 0
        processed = 0
        interrupted = False
        started = next_frame_time = time.perf_counter()

        try:
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Could not read any frame from video: {video_path}")

            self.counters = [build_counter(item, frame.shape) for item in self.config["counters"]]
            if display:
                self._open_window(frame)

            while ok:
                frame_id += 1
                processed += 1

                detections = self.detector.detect(frame)
                for counter in self.counters:
                    known = len(getattr(counter, "events", ()))
                    counter.update(detections)
                    for event in getattr(counter, "events", ())[known:]:
                        event["source_frame"] = frame_id  # frame của video gốc, kể cả khi dùng vid_stride

                # Lưu txt/crop trước khi vẽ, vì draw_boxes vẽ thẳng lên frame.
                saver.save_raw(frame, detections, frame_id)
                annotated = self._annotate(frame, detections)
                saver.save_annotated(annotated, frame_id, output_fps)

                if display:
                    if not self._show(annotated):
                        logger.info("Stopped by user at frame %d", frame_id)
                        interrupted = True
                        break
                    if realtime:
                        # Lịch tuyệt đối thay vì chờ từng frame: tránh trôi và không dồn nợ khi xử lý chậm.
                        # time.sleep chính xác hơn cv2.waitKey (Windows làm tròn waitKey lên ~15 ms).
                        next_frame_time += frame_interval
                        delay = next_frame_time - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                        else:
                            next_frame_time = time.perf_counter()

                for _ in range(vid_stride - 1):
                    if not cap.grab():
                        break
                    frame_id += 1

                ok, frame = cap.read()
        except KeyboardInterrupt:
            logger.info("Interrupted at frame %d", frame_id)
            interrupted = True
        finally:
            cap.release()
            saver.save_events(self._collect_events(source_fps))
            saver.close()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - started
        summary = {
            "video": str(video_path),
            "frames_read": frame_id,
            "frames_processed": processed,
            "fps": processed / elapsed if elapsed > 0 else 0.0,
            "counts": [
                {"type": item["type"], "count": counter.count}
                for item, counter in zip(self.config["counters"], self.counters, strict=True)
            ],
            "output_dir": str(saver.run_dir) if saver.run_dir else None,
            "interrupted": interrupted,
        }
        logger.info(
            "Processed %d frames at %.1f FPS. Final counts: %s",
            processed,
            summary["fps"],
            ", ".join(f"{item['type']}={item['count']}" for item in summary["counts"]),
        )
        return summary
