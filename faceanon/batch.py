"""Batch processing module for face anonymization across directories."""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from .engine import FaceAnonEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class BatchConfig:
    input_dir: str
    output_dir: str
    recursive: bool = True


@dataclass
class FileResult:
    path: str
    status: str  # "success" | "error"
    error: Optional[str] = None
    processing_time: float = 0.0
    faces_detected: int = 0


@dataclass
class BatchReport:
    total_files: int = 0
    successes: int = 0
    failures: int = 0
    results: list = field(default_factory=list)
    total_time: float = 0.0


class BatchProcessor:
    """Scans a directory and batch-processes all supported media files."""

    def __init__(self, engine: FaceAnonEngine, config: BatchConfig):
        self.engine = engine
        self.config = config

    def run(self) -> BatchReport:
        """Process all discovered files. Returns a BatchReport."""
        files = self._discover_files()
        report = BatchReport(total_files=len(files))

        if not files:
            print("No supported files found.")
            return report

        start_time = time.time()

        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="Processing", unit="file")
        except ImportError:
            iterator = files

        for i, (file_path, file_type) in enumerate(iterator):
            rel_path = os.path.relpath(file_path, self.config.input_dir)
            output_path = os.path.join(self.config.output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if file_type == "image":
                result = self._process_image_file(file_path, output_path)
            else:
                result = self._process_video_file(file_path, output_path)

            result.path = rel_path
            report.results.append(result)

            if result.status == "success":
                report.successes += 1
            else:
                report.failures += 1

            if not hasattr(iterator, "set_postfix"):
                if (i + 1) % 10 == 0 or (i + 1) == len(files):
                    print(f"  [{i + 1}/{len(files)}] {report.successes} ok, {report.failures} failed")

        report.total_time = time.time() - start_time
        return report

    def save_report(self, report: BatchReport, path: str) -> None:
        """Write the batch report as JSON."""
        data = {
            "total_files": report.total_files,
            "successes": report.successes,
            "failures": report.failures,
            "total_time_seconds": round(report.total_time, 2),
            "results": [
                {
                    "path": r.path,
                    "status": r.status,
                    "error": r.error,
                    "processing_time": round(r.processing_time, 3),
                    "faces_detected": r.faces_detected,
                }
                for r in report.results
            ],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _discover_files(self) -> list:
        """Walk input_dir and return list of (absolute_path, 'image'|'video')."""
        files = []
        if self.config.recursive:
            walker = os.walk(self.config.input_dir)
        else:
            try:
                top = next(os.walk(self.config.input_dir))
                walker = [top]
            except StopIteration:
                return files

        for root, _, filenames in walker:
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    files.append((os.path.join(root, fname), "image"))
                elif ext in VIDEO_EXTENSIONS:
                    files.append((os.path.join(root, fname), "video"))
        return files

    def _process_image_file(self, input_path: str, output_path: str) -> FileResult:
        start = time.time()
        try:
            image = cv2.imread(input_path)
            if image is None:
                return FileResult(
                    path=input_path, status="error",
                    error="Cannot read image file",
                    processing_time=time.time() - start,
                )
            result = self.engine.process_image(image)
            cv2.imwrite(output_path, result.anonymized_frame)
            return FileResult(
                path=input_path, status="success",
                processing_time=time.time() - start,
                faces_detected=len(result.detections),
            )
        except Exception as e:
            return FileResult(
                path=input_path, status="error",
                error=str(e),
                processing_time=time.time() - start,
            )

    def _process_video_file(self, input_path: str, output_path: str) -> FileResult:
        start = time.time()
        try:
            total_faces = 0

            def callback(idx, total, frame_result):
                nonlocal total_faces
                total_faces += len(frame_result.tracks)

            self.engine.process_video(input_path, output_path, callback=callback)
            return FileResult(
                path=input_path, status="success",
                processing_time=time.time() - start,
                faces_detected=total_faces,
            )
        except Exception as e:
            return FileResult(
                path=input_path, status="error",
                error=str(e),
                processing_time=time.time() - start,
            )
