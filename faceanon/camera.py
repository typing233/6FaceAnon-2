"""Real-time camera processing module for face anonymization."""

import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .engine import FaceAnonEngine


@dataclass
class CameraConfig:
    source: object = 0
    output_path: Optional[str] = None
    display: bool = True
    window_name: str = "FaceAnon"


class CameraProcessor:
    """Captures frames from a camera/stream and processes them in real-time."""

    def __init__(self, engine: FaceAnonEngine, config: Optional[CameraConfig] = None):
        if config is None:
            config = CameraConfig()
        self.engine = engine
        self.config = config

        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_idx = 0
        self._fps_ema = 0.0

    def run(self) -> None:
        """Blocking main loop. Runs until stop() is called or 'q' is pressed."""
        if not self.config.display and not self.config.output_path:
            raise ValueError("Either display or output_path must be set")

        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {self.config.source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if self.config.output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.config.output_path, fourcc, src_fps, (width, height)
            )

        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *_: self.stop())

        reader_thread = threading.Thread(target=self._reader_loop, args=(cap,), daemon=True)
        reader_thread.start()

        try:
            self._process_loop(writer)
        finally:
            self._stop_event.set()
            reader_thread.join(timeout=2.0)
            cap.release()
            if writer:
                writer.release()
            if self.config.display:
                cv2.destroyAllWindows()
            signal.signal(signal.SIGINT, original_sigint)

    def stop(self) -> None:
        """Signal the processor to shut down."""
        self._stop_event.set()

    def _reader_loop(self, cap: cv2.VideoCapture) -> None:
        """Daemon thread that continuously grabs the latest frame."""
        retries = 0
        max_retries = 3

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                retries += 1
                if retries > max_retries:
                    self._stop_event.set()
                    break
                time.sleep(1.0)
                continue
            retries = 0
            with self._frame_lock:
                self._latest_frame = frame

    def _process_loop(self, writer: Optional[cv2.VideoWriter]) -> None:
        """Main thread loop: grab latest frame, process, display/write."""
        self.engine.tracker.reset()
        last_time = time.perf_counter()

        while not self._stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            result = self.engine._process_frame(frame, self._frame_idx)
            self._frame_idx += 1

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            instant_fps = 1.0 / max(dt, 0.001)
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant_fps if self._fps_ema > 0 else instant_fps

            output_frame = result.anonymized_frame

            if writer:
                writer.write(output_frame)

            if self.config.display:
                display_frame = output_frame.copy()
                label = f"FPS: {self._fps_ema:.1f} | Faces: {len(result.tracks)}"
                cv2.putText(
                    display_frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow(self.config.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._stop_event.set()
