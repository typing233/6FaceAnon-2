import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .datatypes import Detection, Track
from .utils import compute_iou_matrix


class KalmanBoxTracker:
    """Kalman filter tracker for a single bounding box."""

    _id_counter = 0

    def __init__(self, bbox: np.ndarray):
        KalmanBoxTracker._id_counter += 1
        self.id = KalmanBoxTracker._id_counter

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # State: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)

        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        self.H = np.eye(4, 8, dtype=np.float64)

        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0

        self.Q = np.eye(8, dtype=np.float64) * 0.01
        self.Q[4:, 4:] *= 0.1

        self.R = np.eye(4, dtype=np.float64) * 1.0

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox()

    def update(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float64)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.state
        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self) -> np.ndarray:
        return self._state_to_bbox()

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.state[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class SORTTracker:
    """Simple Online Realtime Tracker using IoU association and Kalman filter."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.trackers: list[KalmanBoxTracker] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracker with new detections. Returns list of active tracks."""
        # Predict existing trackers
        predicted_boxes = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))

        if len(detections) == 0:
            self._prune_dead()
            return self._get_tracks()

        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32)

        # Match
        matched, unmatched_dets, unmatched_trks = self._associate(
            det_boxes, predicted_boxes
        )

        # Update matched trackers
        for d_idx, t_idx in matched:
            self.trackers[t_idx].update(det_boxes[d_idx])

        # Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(det_boxes[d_idx]))

        self._prune_dead()
        return self._get_tracks()

    def predict_only(self) -> list[Track]:
        """Advance trackers without new detections (for detect_every_n skip frames)."""
        for t in self.trackers:
            t.predict()
            t.time_since_update += 1
            t.age += 1
        # Undo the double-count from predict() itself
        for t in self.trackers:
            t.age -= 1
            t.time_since_update -= 1
        self._prune_dead()
        return self._get_tracks()

    def reset(self):
        self.trackers.clear()
        KalmanBoxTracker._id_counter = 0

    def _associate(
        self, det_boxes: np.ndarray, trk_boxes: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = compute_iou_matrix(det_boxes, trk_boxes)
        cost_matrix = 1.0 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(len(det_boxes)))
        unmatched_trks = set(range(len(trk_boxes)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.config.iou_threshold:
                matched.append((r, c))
                unmatched_dets.discard(r)
                unmatched_trks.discard(c)

        return matched, list(unmatched_dets), list(unmatched_trks)

    def _prune_dead(self):
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.config.max_age
        ]

    def _get_tracks(self) -> list[Track]:
        tracks = []
        for t in self.trackers:
            if t.hits >= self.config.min_hits:
                state = "confirmed"
            elif t.time_since_update > 0:
                state = "lost"
            else:
                state = "tentative"

            tracks.append(Track(
                track_id=t.id,
                bbox=t.get_bbox(),
                state=state,
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
            ))
        return tracks
