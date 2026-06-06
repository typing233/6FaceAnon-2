import numpy as np
from faceanon.tracker import SORTTracker, KalmanBoxTracker
from faceanon.config import TrackerConfig
from faceanon.datatypes import Detection


def _make_det(x1, y1, x2, y2, score=0.9):
    return Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), score=score)


def test_tracker_creates_tracks():
    tracker = SORTTracker(TrackerConfig(min_hits=1))
    dets = [_make_det(10, 10, 50, 50)]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    assert tracks[0].track_id > 0


def test_tracker_maintains_identity():
    tracker = SORTTracker(TrackerConfig(min_hits=1, iou_threshold=0.2))
    dets1 = [_make_det(10, 10, 50, 50)]
    tracks1 = tracker.update(dets1)

    dets2 = [_make_det(12, 12, 52, 52)]
    tracks2 = tracker.update(dets2)

    assert tracks1[0].track_id == tracks2[0].track_id


def test_tracker_separate_tracks_for_distant_faces():
    tracker = SORTTracker(TrackerConfig(min_hits=1))
    dets = [_make_det(0, 0, 30, 30), _make_det(200, 200, 250, 250)]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_tracker_prunes_lost_tracks():
    tracker = SORTTracker(TrackerConfig(min_hits=1, max_age=2))
    tracker.update([_make_det(10, 10, 50, 50)])

    # No detections for 3 frames
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])
    assert len(tracks) == 0


def test_tracker_reset():
    tracker = SORTTracker(TrackerConfig(min_hits=1))
    tracker.update([_make_det(10, 10, 50, 50)])
    tracker.reset()
    assert len(tracker.trackers) == 0


def test_predict_only():
    tracker = SORTTracker(TrackerConfig(min_hits=1))
    tracker.update([_make_det(10, 10, 50, 50)])
    tracks = tracker.predict_only()
    assert len(tracks) == 1


def test_kalman_predict_returns_valid_bbox():
    kt = KalmanBoxTracker(np.array([10, 10, 50, 50], dtype=np.float32))
    bbox = kt.predict()
    assert bbox.shape == (4,)
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]
