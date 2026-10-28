"""Face detection, landmarks, and cropping, per WMamba Appendix B.2.

  * RetinaFace produces bounding boxes.
  * Crop margins differ by phase and this is load-bearing:
      train    : random margin in [4%, 20%] of box size
      inference: fixed 12.5% margin
  * dlib's 81-point predictor supplies landmarks -- TRAINING ONLY (SBI masks).

Detector weights load lazily (heavy, and unit tests must not need them).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int]:
    """Expand (x1, y1, x2, y2) by `margin` (fraction of box W/H per side),
    clipped to image bounds. Returns integer crop coords."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid box {box}")
    bw, bh = x2 - x1, y2 - y1
    x1e = max(int(round(x1 - margin * bw)), 0)
    y1e = max(int(round(y1 - margin * bh)), 0)
    x2e = min(int(round(x2 + margin * bw)), img_w)
    y2e = min(int(round(y2 + margin * bh)), img_h)
    if x2e <= x1e or y2e <= y1e:
        raise ValueError(f"box collapsed after clipping: {box} margin={margin}")
    return x1e, y1e, x2e, y2e


def sample_train_margin(rng: np.random.Generator,
                        lo: float = 0.04, hi: float = 0.20) -> float:
    """[PAPER B.2] random 4-20% training margin."""
    return float(rng.uniform(lo, hi))


EVAL_MARGIN = 0.125  # [PAPER B.2] fixed 12.5% at inference


class FaceDetector:
    """Lazy RetinaFace wrapper. detect() -> list of (box, score), best-first."""

    def __init__(self, device: str = "cuda", confidence: float = 0.7) -> None:
        self.device = device
        self.confidence = confidence
        self._model = None

    def _ensure(self):
        if self._model is None:
            from retinaface.pre_trained_models import get_model
            self._model = get_model("resnet50_2020-07-20", max_size=1024,
                                    device=self.device)
            self._model.eval()
        return self._model

    def detect(self, image_rgb: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        model = self._ensure()
        anns = model.predict_jsons(image_rgb)
        out = []
        for a in anns:
            if a.get("score", 0) >= self.confidence and a.get("bbox"):
                x1, y1, x2, y2 = a["bbox"]
                if x2 > x1 and y2 > y1:
                    out.append(((float(x1), float(y1), float(x2), float(y2)),
                                float(a["score"])))
        out.sort(key=lambda t: -t[1])
        return out


class LandmarkPredictor:
    """Lazy dlib 81-point landmark predictor (training/SBI only)."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._detector = None
        self._predictor = None

    def _ensure(self):
        if self._predictor is None:
            import dlib
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"dlib 81-point predictor missing: {self.model_path}"
                )
            self._detector = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(str(self.model_path))
        return self._detector, self._predictor

    def landmarks(self, image_rgb: np.ndarray,
                  box: tuple[float, float, float, float] | None = None) -> np.ndarray:
        """(81, 2) float32 pixel coords. Raises if no face/landmarks found."""
        import dlib
        detector, predictor = self._ensure()
        if box is not None:
            rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        else:
            dets = detector(image_rgb, 1)
            if len(dets) == 0:
                raise RuntimeError("dlib found no face for landmarking")
            rect = dets[0]
        shape = predictor(image_rgb, rect)
        pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        if pts.shape != (81, 2):
            raise RuntimeError(f"expected 81 landmarks, got {pts.shape}")
        return pts
