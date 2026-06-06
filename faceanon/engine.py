from typing import Optional, Callable, Iterator
import numpy as np
import cv2

from .config import EngineConfig
from .datatypes import Detection, Track, FrameResult
from .detector import CenterFaceDetector
from .tracker import SORTTracker
from .anonymizer import Anonymizer


class FaceAnonEngine:
    """Main orchestrator: detect faces, track them, and apply anonymization."""

    def __init__(self, config: Optional[EngineConfig] = None):
        if config is None:
            config = EngineConfig()
        self.config = config

        self.detector = CenterFaceDetector(config.detector)
        self.tracker = SORTTracker(config.tracker)
        self.anonymizer = Anonymizer(config.anonymizer)

    def process_image(self, image: np.ndarray) -> FrameResult:
        """Process a single BGR image. Returns FrameResult with anonymized frame."""
        detections = self.detector.detect(image)
        tracks = self.tracker.update(detections)

        # For single images, anonymize all detected faces regardless of track state
        all_tracks = []
        for det in detections:
            all_tracks.append(Track(
                track_id=0,
                bbox=det.bbox,
                state="confirmed",
            ))

        anonymized = self.anonymizer.anonymize(image, all_tracks)
        return FrameResult(
            frame_index=0,
            detections=detections,
            tracks=all_tracks,
            anonymized_frame=anonymized,
        )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        callback: Optional[Callable[[int, int, FrameResult], None]] = None,
    ) -> int:
        """Process a video file end-to-end. Returns total frames processed."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        self.tracker.reset()
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self._process_frame(frame, frame_idx)
                writer.write(result.anonymized_frame)

                if callback:
                    callback(frame_idx, total_frames, result)

                frame_idx += 1
        finally:
            cap.release()
            writer.release()

        return frame_idx

    def process_video_frames(
        self, frames: Iterator[np.ndarray]
    ) -> Iterator[FrameResult]:
        """Generator-based processing of a frame iterator."""
        self.tracker.reset()
        for idx, frame in enumerate(frames):
            yield self._process_frame(frame, idx)

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        run_detection = (frame_idx % self.config.detect_every_n == 0)

        if run_detection:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
        else:
            detections = []
            tracks = self.tracker.predict_only()

        anonymized = self.anonymizer.anonymize(frame, tracks)

        return FrameResult(
            frame_index=frame_idx,
            detections=detections,
            tracks=tracks,
            anonymized_frame=anonymized,
        )
