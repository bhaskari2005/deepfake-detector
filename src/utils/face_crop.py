"""
Face detection + landmark-based alignment, replicating DeepfakeBench's
own preprocessing/preprocess.py (img_align_crop) exactly, but using
MTCNN's 5-point landmarks instead of dlib's. This matters: these models
were trained on faces warped so eyes/nose/mouth land at fixed canonical
pixel positions every time, not just a centered box crop.
"""
import numpy as np
import cv2
from skimage import transform as trans
from PIL import Image
from facenet_pytorch import MTCNN

# ArcFace-style canonical 5-point template (eyes, nose, mouth corners)
# at 112x112 reference scale — taken directly from DeepfakeBench's code.
_CANONICAL_5PTS_112 = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)


def _build_dst(res, scale=1.3):
    dst = _CANONICAL_5PTS_112.copy()
    dst[:, 0] += 8.0
    dst[:, 0] = dst[:, 0] * res / 112.0
    dst[:, 1] = dst[:, 1] * res / 112.0

    margin_rate = scale - 1
    x_margin = res * margin_rate / 2.0
    y_margin = res * margin_rate / 2.0
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= res / (res + 2 * x_margin)
    dst[:, 1] *= res / (res + 2 * y_margin)
    return dst


def align_face(image: Image.Image, landmarks: np.ndarray, res: int, scale: float = 1.3) -> Image.Image:
    """
    landmarks: 5x2 array, order = [left_eye, right_eye, nose, mouth_left, mouth_right]
    (MTCNN's native landmark order — same order DeepfakeBench used from dlib).
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    dst = _build_dst(res, scale)
    src = landmarks.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    aligned_bgr = cv2.warpAffine(img_bgr, M, (res, res))
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(aligned_rgb)


class FaceCropper:
    def __init__(self, device="cuda", min_confidence=0.90):
        self.min_confidence = min_confidence
        self.mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image: Image.Image):
        """
        Returns list of {box, confidence, landmarks} in ORIGINAL image
        coordinates. Alignment happens later, per-detector, via align_face().
        """
        image = image.convert("RGB")
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        if boxes is None:
            return []

        results = []
        for box, prob, lm in zip(boxes, probs, landmarks):
            if prob is None or prob < self.min_confidence:
                continue
            results.append({
                "box": tuple(int(v) for v in box),
                "confidence": float(prob),
                "landmarks": np.array(lm, dtype=np.float32),
            })
        return results
