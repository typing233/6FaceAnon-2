import numpy as np
from faceanon.utils import compute_iou_matrix, nms


def test_iou_identical_boxes():
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    iou = compute_iou_matrix(boxes, boxes)
    assert iou.shape == (1, 1)
    assert abs(iou[0, 0] - 1.0) < 1e-5


def test_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    iou = compute_iou_matrix(a, b)
    assert iou[0, 0] == 0.0


def test_iou_partial_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 5, 15, 15]], dtype=np.float32)
    iou = compute_iou_matrix(a, b)
    # Intersection = 5*5=25, Union = 100+100-25=175
    expected = 25.0 / 175.0
    assert abs(iou[0, 0] - expected) < 1e-5


def test_iou_empty_input():
    a = np.empty((0, 4), dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = compute_iou_matrix(a, b)
    assert iou.shape == (0, 1)


def test_nms_empty():
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.array([], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert len(keep) == 0


def test_nms_single():
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert len(keep) == 1
    assert keep[0] == 0


def test_nms_suppresses_overlap():
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = nms(boxes, scores, 0.3)
    assert len(keep) == 1
    assert keep[0] == 0


def test_nms_keeps_separate():
    boxes = np.array([
        [0, 0, 10, 10],
        [100, 100, 110, 110],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert len(keep) == 2
