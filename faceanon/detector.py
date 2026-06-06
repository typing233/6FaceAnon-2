import numpy as np
import cv2
import onnxruntime as ort

from .config import DetectorConfig
from .datatypes import Detection
from .utils import nms, ensure_model


class CenterFaceDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config

        model_path = ensure_model(config.model_path)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.enable_mem_pattern = False

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect faces in a BGR image. Returns list of Detection objects."""
        img_h, img_w = image.shape[:2]
        inp_h, inp_w = self.config.input_size

        blob = self._preprocess(image, inp_h, inp_w)
        outputs = self.session.run(None, {self.input_name: blob})

        detections = self._postprocess(outputs, img_h, img_w, inp_h, inp_w)
        return detections

    def _preprocess(self, image: np.ndarray, inp_h: int, inp_w: int) -> np.ndarray:
        resized = cv2.resize(image, (inp_w, inp_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        return blob

    def _postprocess(
        self, outputs: list, img_h: int, img_w: int, inp_h: int, inp_w: int
    ) -> list[Detection]:
        heatmap, scale, offset = outputs[0], outputs[1], outputs[2]
        landmarks_raw = outputs[3] if len(outputs) > 3 else None

        heatmap = heatmap[0, 0]
        scale0, scale1 = scale[0, 0], scale[0, 1]
        offset0, offset1 = offset[0, 0], offset[0, 1]

        feat_h, feat_w = heatmap.shape
        scale_h = img_h / feat_h
        scale_w = img_w / feat_w

        ys, xs = np.where(heatmap > self.config.score_threshold)
        if len(ys) == 0:
            return []

        scores = heatmap[ys, xs]

        cx = (xs + offset1[ys, xs]) * scale_w
        cy = (ys + offset0[ys, xs]) * scale_h
        w = scale1[ys, xs] * scale_w
        h = scale0[ys, xs] * scale_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1 = np.clip(x1, 0, img_w)
        y1 = np.clip(y1, 0, img_h)
        x2 = np.clip(x2, 0, img_w)
        y2 = np.clip(y2, 0, img_h)

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes, scores, self.config.nms_threshold)

        detections = []
        for idx in keep:
            lms = None
            if landmarks_raw is not None:
                lms_data = landmarks_raw[0, :, ys[idx], xs[idx]]
                lms = np.zeros((5, 2), dtype=np.float32)
                for j in range(5):
                    lms[j, 0] = (lms_data[j * 2 + 1] + xs[idx]) * scale_w
                    lms[j, 1] = (lms_data[j * 2] + ys[idx]) * scale_h

            detections.append(Detection(
                bbox=boxes[idx],
                score=float(scores[idx]),
                landmarks=lms,
            ))

        return detections
