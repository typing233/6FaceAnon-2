from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os


class AnonymizationType(Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    MOSAIC = "mosaic"


@dataclass
class DetectorConfig:
    model_path: Optional[str] = None
    score_threshold: float = 0.5
    nms_threshold: float = 0.3
    input_size: tuple[int, int] = (640, 640)

    def __post_init__(self):
        if self.model_path is None:
            self.model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "models", "centerface.onnx"
            )
            self.model_path = os.path.normpath(self.model_path)


@dataclass
class TrackerConfig:
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class AnonymizerConfig:
    method: AnonymizationType = AnonymizationType.GAUSSIAN_BLUR
    intensity: float = 0.8
    expand_ratio: float = 0.1
    elliptical_mask: bool = True


@dataclass
class EngineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    detect_every_n: int = 1
