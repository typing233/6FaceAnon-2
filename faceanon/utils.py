import numpy as np
import os
import urllib.request
import sys

MODEL_URL = "https://raw.githubusercontent.com/Star-Clouds/CenterFace/master/models/onnx/centerface_bnmerged.onnx"


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes. Each box is [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    return (intersection / np.maximum(union, 1e-6)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Non-maximum suppression. Returns indices of kept boxes."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def get_bundled_model_path() -> str:
    """Resolve model path, checking PyInstaller bundle first."""
    if getattr(sys, '_MEIPASS', None):
        return os.path.join(sys._MEIPASS, 'models', 'centerface.onnx')
    return os.path.join(os.path.dirname(__file__), '..', 'models', 'centerface.onnx')


def ensure_model(model_path: str) -> str:
    """Ensure the CenterFace ONNX model exists, downloading if necessary."""
    if os.path.isfile(model_path):
        return model_path

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Downloading CenterFace model to {model_path}...", file=sys.stderr)

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / max(total_size, 1))
        sys.stderr.write(f"\r  {min(percent, 100)}%")
        sys.stderr.flush()

    urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_progress)
    sys.stderr.write("\n")

    _fix_dynamic_input(model_path)
    return model_path


def _fix_dynamic_input(model_path: str):
    """Patch the ONNX model: remove initializers from inputs and set dynamic dims."""
    try:
        import onnx

        model = onnx.load(model_path)

        # Remove initializers from graph inputs (they shouldn't be there)
        initializer_names = {init.name for init in model.graph.initializer}
        new_inputs = []
        for inp in model.graph.input:
            if inp.name not in initializer_names:
                new_inputs.append(inp)
        del model.graph.input[:]
        model.graph.input.extend(new_inputs)

        # Set dynamic input dimensions on the actual image input
        if len(model.graph.input) > 0:
            inp = model.graph.input[0]
            shape = inp.type.tensor_type.shape
            if len(shape.dim) >= 4:
                for dim in shape.dim:
                    dim.Clear()
                shape.dim[0].dim_param = "batch"
                shape.dim[1].dim_value = 3
                shape.dim[2].dim_param = "height"
                shape.dim[3].dim_param = "width"

        onnx.save(model, model_path)
    except ImportError:
        pass
