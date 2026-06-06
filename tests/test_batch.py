"""Tests for the batch processing module."""

import json
import os
import tempfile

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from faceanon.batch import (
    BatchConfig,
    BatchProcessor,
    BatchReport,
    FileResult,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from faceanon.config import EngineConfig
from faceanon.datatypes import FrameResult, Detection, Track


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig(input_dir="/in", output_dir="/out")
        assert cfg.recursive is True

    def test_non_recursive(self):
        cfg = BatchConfig(input_dir="/in", output_dir="/out", recursive=False)
        assert cfg.recursive is False


class TestBatchProcessor:
    def _make_engine_mock(self):
        engine = MagicMock()
        fake_result = FrameResult(
            frame_index=0,
            detections=[Detection(bbox=np.array([10, 10, 50, 50]), score=0.9)],
            tracks=[Track(track_id=1, bbox=np.array([10, 10, 50, 50]), state="confirmed")],
            anonymized_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        )
        engine.process_image.return_value = fake_result
        engine.process_video.return_value = 30
        return engine

    def test_discover_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".jpg", ".png", ".bmp"]:
                open(os.path.join(tmpdir, f"test{ext}"), "w").close()
            open(os.path.join(tmpdir, "readme.txt"), "w").close()

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir="/tmp/out")
            processor = BatchProcessor(engine, cfg)
            files = processor._discover_files()

            assert len(files) == 3
            assert all(ft == "image" for _, ft in files)

    def test_discover_videos(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".mp4", ".avi"]:
                open(os.path.join(tmpdir, f"vid{ext}"), "w").close()

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir="/tmp/out")
            processor = BatchProcessor(engine, cfg)
            files = processor._discover_files()

            assert len(files) == 2
            assert all(ft == "video" for _, ft in files)

    def test_discover_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            open(os.path.join(tmpdir, "a.jpg"), "w").close()
            open(os.path.join(subdir, "b.png"), "w").close()

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir="/tmp/out", recursive=True)
            processor = BatchProcessor(engine, cfg)
            files = processor._discover_files()
            assert len(files) == 2

    def test_discover_non_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub")
            os.makedirs(subdir)
            open(os.path.join(tmpdir, "a.jpg"), "w").close()
            open(os.path.join(subdir, "b.png"), "w").close()

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir="/tmp/out", recursive=False)
            processor = BatchProcessor(engine, cfg)
            files = processor._discover_files()
            assert len(files) == 1

    def test_process_image_file_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            input_path = os.path.join(tmpdir, "test.jpg")
            output_path = os.path.join(tmpdir, "out.jpg")
            cv2.imwrite(input_path, img)

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir=tmpdir)
            processor = BatchProcessor(engine, cfg)
            result = processor._process_image_file(input_path, output_path)

            assert result.status == "success"
            assert result.faces_detected == 1

    def test_process_image_file_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "corrupt.jpg")
            output_path = os.path.join(tmpdir, "out.jpg")
            with open(input_path, "w") as f:
                f.write("not an image")

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir=tmpdir, output_dir=tmpdir)
            processor = BatchProcessor(engine, cfg)
            result = processor._process_image_file(input_path, output_path)

            assert result.status == "error"
            assert result.error is not None

    def test_save_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = BatchReport(
                total_files=3,
                successes=2,
                failures=1,
                results=[
                    FileResult(path="a.jpg", status="success", processing_time=0.5, faces_detected=2),
                    FileResult(path="b.png", status="success", processing_time=0.3, faces_detected=1),
                    FileResult(path="c.jpg", status="error", error="Cannot read", processing_time=0.01),
                ],
                total_time=0.81,
            )

            engine = self._make_engine_mock()
            cfg = BatchConfig(input_dir="/in", output_dir="/out")
            processor = BatchProcessor(engine, cfg)

            report_path = os.path.join(tmpdir, "report.json")
            processor.save_report(report, report_path)

            with open(report_path) as f:
                data = json.load(f)

            assert data["total_files"] == 3
            assert data["successes"] == 2
            assert data["failures"] == 1
            assert len(data["results"]) == 3

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = self._make_engine_mock()
            outdir = os.path.join(tmpdir, "out")
            cfg = BatchConfig(input_dir=tmpdir, output_dir=outdir)
            processor = BatchProcessor(engine, cfg)
            report = processor.run()

            assert report.total_files == 0
            assert report.successes == 0
            assert report.failures == 0
