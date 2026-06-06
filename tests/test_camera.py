"""Tests for the camera module."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from faceanon.camera import CameraConfig, CameraProcessor
from faceanon.config import EngineConfig


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.source == 0
        assert cfg.output_path is None
        assert cfg.display is True
        assert cfg.window_name == "FaceAnon"

    def test_custom_source(self):
        cfg = CameraConfig(source="rtsp://example.com/stream")
        assert cfg.source == "rtsp://example.com/stream"

    def test_output_path(self):
        cfg = CameraConfig(output_path="/tmp/out.mp4", display=False)
        assert cfg.output_path == "/tmp/out.mp4"
        assert cfg.display is False


class TestCameraProcessor:
    def _make_engine_mock(self):
        engine = MagicMock()
        engine.tracker = MagicMock()
        fake_result = MagicMock()
        fake_result.anonymized_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_result.tracks = []
        engine._process_frame.return_value = fake_result
        return engine

    def test_no_display_no_output_raises(self):
        engine = self._make_engine_mock()
        cfg = CameraConfig(display=False, output_path=None)
        processor = CameraProcessor(engine, cfg)
        with pytest.raises(ValueError, match="Either display or output_path"):
            processor.run()

    def test_stop_event(self):
        engine = self._make_engine_mock()
        cfg = CameraConfig(display=False, output_path="/tmp/test_out.mp4")
        processor = CameraProcessor(engine, cfg)
        processor._stop_event.set()
        assert processor._stop_event.is_set()

    @patch("faceanon.camera.cv2")
    def test_process_loop_stops_on_event(self, mock_cv2):
        engine = self._make_engine_mock()
        cfg = CameraConfig(display=False, output_path="/tmp/test.mp4")
        processor = CameraProcessor(engine, cfg)

        processor._stop_event.set()
        processor._process_loop(None)
        engine._process_frame.assert_not_called()
