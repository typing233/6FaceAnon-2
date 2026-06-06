import numpy as np
import cv2

from .config import AnonymizerConfig, AnonymizationType
from .datatypes import Track


class Anonymizer:
    def __init__(self, config: AnonymizerConfig):
        self.config = config

    def anonymize(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Apply anonymization to all tracked face regions in the frame."""
        result = frame.copy()
        for track in tracks:
            self._anonymize_face(result, track.bbox)
        return result

    def _anonymize_face(self, frame: np.ndarray, bbox: np.ndarray):
        img_h, img_w = frame.shape[:2]

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        # Expand bbox
        ex = w * self.config.expand_ratio
        ey = h * self.config.expand_ratio
        x1 = int(max(0, x1 - ex))
        y1 = int(max(0, y1 - ey))
        x2 = int(min(img_w, x2 + ex))
        y2 = int(min(img_h, y2 + ey))

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w < 2 or roi_h < 2:
            return

        roi = frame[y1:y2, x1:x2]

        if self.config.method == AnonymizationType.GAUSSIAN_BLUR:
            processed = self._apply_blur(roi)
        else:
            processed = self._apply_mosaic(roi)

        if self.config.elliptical_mask:
            mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
            center = (roi_w // 2, roi_h // 2)
            axes = (roi_w // 2, roi_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask_3ch = mask[:, :, np.newaxis] / 255.0
            blended = (processed * mask_3ch + roi * (1.0 - mask_3ch)).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
        else:
            frame[y1:y2, x1:x2] = processed

    def _apply_blur(self, roi: np.ndarray) -> np.ndarray:
        face_size = min(roi.shape[0], roi.shape[1])
        ksize = int(self.config.intensity * face_size * 0.8)
        ksize = max(3, ksize)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 99)
        return cv2.GaussianBlur(roi, (ksize, ksize), 0)

    def _apply_mosaic(self, roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        face_size = min(h, w)
        block_size = max(2, int(self.config.intensity * face_size * 0.3))
        small_w = max(1, w // block_size)
        small_h = max(1, h // block_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
