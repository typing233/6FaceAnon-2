"""Integration tests for FaceAnonEngine (require model download)."""
import numpy as np
import pytest
import os
import tempfile
import cv2

from faceanon import FaceAnonEngine, EngineConfig, DetectorConfig


@pytest.fixture(scope="module")
def engine():
    return FaceAnonEngine(EngineConfig())


def test_process_image_blank(engine):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.process_image(blank)
    assert result.anonymized_frame.shape == blank.shape
    assert len(result.detections) == 0


def test_process_image_returns_frame_result(engine):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = engine.process_image(img)
    assert result.frame_index == 0
    assert result.anonymized_frame is not None
    assert result.anonymized_frame.shape == img.shape


def test_process_video_frames_generator(engine):
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    results = list(engine.process_video_frames(iter(frames)))
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.frame_index == i


def test_process_video_file(engine):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "test_input.mp4")
        output_path = os.path.join(tmpdir, "test_output.mp4")

        # Create a short synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_path, fourcc, 25.0, (640, 480))
        for _ in range(10):
            writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.release()

        total = engine.process_video(input_path, output_path)
        assert total == 10
        assert os.path.isfile(output_path)
