"""Kiểm tra logic đếm bằng dữ liệu giả, không cần model. Chạy: py -m unittest discover -s tests -t ."""
import unittest

import numpy as np

from src.counter import LaneZoneCounter, LineCounter, ZoneCounter


def det(track_id, x, y):
    """Detection giả có tâm tại (x, y)."""
    return {"bbox": (x - 10, y - 10, x + 10, y + 10), "track_id": track_id, "conf": 0.9, "class": 0}


def play(counter, *positions, track_id=1):
    for position in positions:
        counter.update([] if position is None else [det(track_id, *position)])
    return counter.count


class LineCounterTest(unittest.TestCase):
    def test_horizontal_line_counts_crossing_once(self):
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (50, 80), (50, 90), (50, 110), (50, 120)), 1)

    def test_no_crossing_is_not_counted(self):
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (50, 10), (50, 50), (50, 90)), 0)

    def test_jitter_around_line_counts_once(self):
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (50, 90), (50, 110), (50, 90), (50, 110)), 1)

    def test_diagonal_line_uses_both_endpoints(self):
        # Với đường chéo này, code cũ (chỉ so với y của start) sẽ không đếm: y luôn lớn hơn start[1].
        counter = LineCounter((0, 0), (200, 200))
        self.assertEqual(play(counter, (120, 80), (80, 120)), 1)

    def test_duplicate_box_on_same_object_counted_once_when_enabled(self):
        # Hai track khác ID, box gần trùng nhau (IoU ~0.8), cùng vượt đường ở cùng frame.
        def cross(counter):
            for y in (90, 110):
                counter.update([det(1, 50, y), {**det(2, 52, y), "bbox": (42, y - 10, 62, y + 10)}])
            return counter.count

        self.assertEqual(cross(LineCounter((0, 100), (200, 100))), 2)
        counter = LineCounter((0, 100), (200, 100), duplicate_iou=0.6)
        self.assertEqual(cross(counter), 1)
        self.assertEqual(counter.duplicates_skipped, 1)

    def test_separate_objects_are_not_treated_as_duplicates(self):
        counter = LineCounter((0, 100), (200, 100), duplicate_iou=0.6)
        for y in (90, 110):
            counter.update([det(1, 30, y), det(2, 120, y)])
        self.assertEqual(counter.count, 2)

    def test_events_record_each_count(self):
        counter = LineCounter((0, 100), (200, 100))
        play(counter, (50, 90), (50, 110))
        self.assertEqual([(e["frame"], e["track_id"]) for e in counter.events], [(2, 1)])

    def test_vertical_line(self):
        counter = LineCounter((100, 0), (100, 200))
        self.assertEqual(play(counter, (90, 50), (110, 50)), 1)

    def test_crossing_outside_segment_is_ignored(self):
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (300, 90), (300, 110)), 0)

    def test_direction_filter(self):
        # Đường vẽ từ trái sang phải: phía dương ở bên dưới.
        down = [(50, 90), (50, 110)]
        up = [(50, 110), (50, 90)]
        self.assertEqual(play(LineCounter((0, 100), (200, 100), direction="positive"), *down), 1)
        self.assertEqual(play(LineCounter((0, 100), (200, 100), direction="positive"), *up), 0)
        self.assertEqual(play(LineCounter((0, 100), (200, 100), direction="negative"), *up), 1)
        self.assertEqual(play(LineCounter((0, 100), (200, 100), direction="negative"), *down), 0)

    def test_center_exactly_on_line_is_handled(self):
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (50, 90), (50, 100), (50, 110)), 1)
        counter = LineCounter((0, 100), (200, 100))
        self.assertEqual(play(counter, (50, 90), (50, 100), (50, 90)), 0)

    def test_track_missing_briefly_is_still_counted(self):
        counter = LineCounter((0, 100), (200, 100), max_missing_frames=5)
        self.assertEqual(play(counter, (50, 90), None, None, (50, 110)), 1)

    def test_state_expires_after_max_missing_frames(self):
        counter = LineCounter((0, 100), (200, 100), max_missing_frames=2)
        self.assertEqual(play(counter, (50, 90), None, None, None, (50, 110)), 0)
        self.assertEqual(counter._tracks[1].side, 1)

    def test_two_tracks_are_counted_separately(self):
        counter = LineCounter((0, 100), (200, 100))
        counter.update([det(1, 50, 90), det(2, 150, 90)])
        counter.update([det(1, 50, 110), det(2, 150, 110)])
        self.assertEqual(counter.count, 2)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            LineCounter((1, 1), (1, 1))
        with self.assertRaises(ValueError):
            LineCounter((0, 0), (1, 1), direction="sideways")


class LaneZoneCounterTest(unittest.TestCase):
    SQUARE = [(100, 100), (200, 100), (200, 200), (100, 200)]
    SHAPE = (300, 300, 3)

    def test_each_track_counted_once(self):
        counter = LaneZoneCounter(self.SQUARE, self.SHAPE)
        self.assertEqual(play(counter, (150, 150), (160, 160), (50, 50), (150, 150)), 1)

    def test_min_frames_rejects_short_tracks(self):
        counter = LaneZoneCounter(self.SQUARE, self.SHAPE, min_frames=3)
        self.assertEqual(play(counter, (150, 150), (150, 150)), 0)
        self.assertEqual(play(counter, (150, 150)), 1)

    def test_multiple_polygons(self):
        other = [(0, 0), (50, 0), (50, 50), (0, 50)]
        counter = LaneZoneCounter([self.SQUARE, other], self.SHAPE)
        counter.update([det(1, 150, 150), det(2, 25, 25), det(3, 250, 250)])
        self.assertEqual(counter.count, 2)

    def test_center_outside_frame_does_not_wrap_around(self):
        # Vùng chạm góc dưới-phải: tọa độ âm không được lấy nhầm điểm ở cuối mảng.
        corner = [(250, 250), (299, 250), (299, 299), (250, 299)]
        counter = LaneZoneCounter(corner, self.SHAPE)
        self.assertEqual(play(counter, (-10, -10), (500, 500)), 0)

    def test_colors_variants(self):
        self.assertEqual(len(LaneZoneCounter(self.SQUARE, self.SHAPE, colors=[]).colors), 1)
        self.assertEqual(LaneZoneCounter(self.SQUARE, self.SHAPE, colors=[1, 2, 3]).colors, [(1, 2, 3)])
        self.assertEqual(LaneZoneCounter(self.SQUARE, self.SHAPE, colors=[[1, 2, 3]]).colors, [(1, 2, 3)])


class ZoneCounterTest(unittest.TestCase):
    def test_counts_only_current_objects(self):
        counter = ZoneCounter((100, 100), (200, 200))
        self.assertEqual(play(counter, (150, 150)), 1)
        self.assertEqual(play(counter, (50, 50)), 0)


class DrawTest(unittest.TestCase):
    def test_draw_before_update_does_not_fail(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        counters = [
            LineCounter((0, 100), (200, 100)),
            ZoneCounter((10, 10), (50, 50)),
            LaneZoneCounter([(1, 1), (9, 1), (9, 9)], frame.shape),
        ]
        for slot, counter in enumerate(counters):
            self.assertEqual(counter.draw(frame.copy(), slot=slot).shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
