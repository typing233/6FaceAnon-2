import numpy as np
from faceanon.anonymizer import Anonymizer
from faceanon.config import AnonymizerConfig, AnonymizationType
from faceanon.datatypes import Track


def _make_track(x1, y1, x2, y2):
    return Track(track_id=1, bbox=np.array([x1, y1, x2, y2], dtype=np.float32), state="confirmed")


def _make_frame(h=200, w=200):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def test_gaussian_blur_modifies_roi():
    config = AnonymizerConfig(method=AnonymizationType.GAUSSIAN_BLUR, intensity=0.8)
    anon = Anonymizer(config)
    frame = _make_frame()
    original = frame.copy()
    track = _make_track(50, 50, 150, 150)
    result = anon.anonymize(frame, [track])
    # The original should not be mutated
    assert np.array_equal(frame, original)
    # The result ROI should differ from original
    roi_orig = original[50:150, 50:150]
    roi_anon = result[50:150, 50:150]
    assert not np.array_equal(roi_orig, roi_anon)


def test_mosaic_modifies_roi():
    config = AnonymizerConfig(method=AnonymizationType.MOSAIC, intensity=0.8)
    anon = Anonymizer(config)
    frame = _make_frame()
    track = _make_track(50, 50, 150, 150)
    result = anon.anonymize(frame, [track])
    roi_orig = frame[50:150, 50:150]
    roi_anon = result[50:150, 50:150]
    assert not np.array_equal(roi_orig, roi_anon)


def test_elliptical_mask_leaves_corners():
    config = AnonymizerConfig(
        method=AnonymizationType.GAUSSIAN_BLUR,
        intensity=0.8,
        elliptical_mask=True,
        expand_ratio=0.0,
    )
    anon = Anonymizer(config)
    frame = np.zeros((200, 200, 3), dtype=np.uint8) + 128
    track = _make_track(50, 50, 150, 150)
    result = anon.anonymize(frame, [track])
    # Corners of the bbox should remain unchanged (outside ellipse)
    assert np.array_equal(result[50, 50], frame[50, 50])
    assert np.array_equal(result[50, 149], frame[50, 149])


def test_no_tracks_returns_unchanged():
    config = AnonymizerConfig()
    anon = Anonymizer(config)
    frame = _make_frame()
    result = anon.anonymize(frame, [])
    assert np.array_equal(result, frame)


def test_intensity_affects_blur():
    frame = _make_frame(100, 100)
    track = _make_track(10, 10, 90, 90)

    low = Anonymizer(AnonymizerConfig(intensity=0.2))
    high = Anonymizer(AnonymizerConfig(intensity=0.9))

    result_low = low.anonymize(frame, [track])
    result_high = high.anonymize(frame, [track])

    # Higher intensity should produce more difference from original
    diff_low = np.abs(result_low.astype(float) - frame.astype(float)).mean()
    diff_high = np.abs(result_high.astype(float) - frame.astype(float)).mean()
    assert diff_high > diff_low
