"""Chạy cả pipeline trên video giả với detector giả, không cần model. Chạy: py -m unittest discover -s tests -t ."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.pipeline import DetectionPipeline
from src.saver.saver import ResultSaver


class FakeDetector:
    """Một vật đi từ trên xuống, mỗi lần gọi detect dịch xuống 20 px."""

    def __init__(self, **kwargs):
        self.model = mock.Mock(names={0: "item"})
        self.calls = 0

    def detect(self, frame):
        y = 10 + 20 * self.calls
        self.calls += 1
        return [{"bbox": (40.0, y - 5.0, 60.0, y + 5.0), "conf": 0.9, "class": 0, "track_id": 1}]


def make_video(path, frames=10, size=(100, 100)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class PipelineTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.video = self.dir / "demo.mp4"
        make_video(self.video)

    def pipeline(self, **saver):
        config = {
            "data": {"output": str(self.dir / "outputs"), "model": str(self.dir)},
            "detector": {"type": "fake", "conf_threshold": 0.1, "iou_threshold": 0.7},
            "tracker": {"config": "unused.yaml"},
            "saver": saver,
            "counters": [
                {"type": "line", "start": [0, 50], "end": [100, 50]},
                {"type": "zone", "top_left": [0, 0], "bottom_right": [99, 99]},
            ],
        }
        with mock.patch("src.pipeline.YOLODetector", FakeDetector):
            return DetectionPipeline(config)

    def test_full_run_counts_and_returns_summary(self):
        summary = self.pipeline().run_video(self.video, display=False)
        self.assertEqual(summary["frames_processed"], 10)
        self.assertEqual(summary["counts"][0], {"type": "line", "count": 1})
        self.assertEqual(len(summary["counts"]), 2)  # chạy đủ mọi counter, không chỉ counter cuối
        self.assertIsNone(summary["output_dir"])
        self.assertFalse((self.dir / "outputs").exists())

    def test_stride_skips_frames_without_feeding_stale_detections(self):
        pipeline = self.pipeline()
        summary = pipeline.run_video(self.video, vid_stride=3, display=False)
        self.assertEqual(summary["frames_processed"], 4)
        self.assertEqual(summary["frames_read"], 10)
        self.assertEqual(pipeline.detector.calls, 4)

    def test_txt_and_crops_are_saved_without_save_images(self):
        # Hồi quy: trước đây save_detections và save_crops bị bỏ qua nếu save_images tắt.
        summary = self.pipeline(save_detections=True, save_crops=True).run_video(self.video, display=False)
        output = Path(summary["output_dir"])
        self.assertEqual(len(list(output.glob("frame_*.txt"))), 10)
        self.assertGreater(len(list(output.glob("crop_*.jpg"))), 0)
        self.assertEqual(list(output.glob("frame_*.jpg")), [])

    def test_saved_video_has_all_frames(self):
        summary = self.pipeline(save_videos=True).run_video(self.video, display=False)
        capture = cv2.VideoCapture(str(Path(summary["output_dir"]) / "result.mp4"))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        capture.release()

    def test_events_csv_lists_each_count_with_time(self):
        summary = self.pipeline(save_videos=True).run_video(self.video, display=False)
        lines = (Path(summary["output_dir"]) / "events.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "counter,n,frame,seconds,track_id,x,y")
        self.assertEqual(len(lines), 2)  # tiêu đề và đúng một lần đếm
        counter, n, frame, seconds, track_id, _, _ = lines[1].split(",")
        self.assertEqual((counter, n, track_id), ("0", "1", "1"))
        self.assertAlmostEqual(float(seconds), int(frame) / 10, places=1)  # video giả chạy 10 FPS

    def test_unreadable_video_raises(self):
        broken = self.dir / "broken.mp4"
        broken.write_bytes(b"not a video")
        with self.assertRaises(IOError):
            self.pipeline().run_video(broken, display=False)


class SaverTest(unittest.TestCase):
    def test_empty_crop_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = ResultSaver(directory, save_crop=True)
            frame = np.zeros((50, 50, 3), dtype=np.uint8)
            detections = [
                {"bbox": (60.0, 60.0, 80.0, 80.0), "track_id": 1, "conf": 0.5, "class": 0},  # ngoài frame
                {"bbox": (10.0, 10.0, 20.0, 20.0), "track_id": 2, "conf": 0.5, "class": 0},
            ]
            saver.save_raw(frame, detections, 1)
            names = [path.name for path in saver.run_dir.iterdir()]
            self.assertEqual(names, ["crop_00001_id2.jpg"])


if __name__ == "__main__":
    unittest.main()
