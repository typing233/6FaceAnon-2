from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray  # shape (4,) [x1, y1, x2, y2]
    score: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2) if available


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray  # shape (4,) [x1, y1, x2, y2]
    state: str = "tentative"  # tentative, confirmed, lost
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


@dataclass
class FrameResult:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    anonymized_frame: Optional[np.ndarray] = None
