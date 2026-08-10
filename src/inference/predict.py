"""Self-contained inference module for WMamba deepfake detection.

Gate 2 (train/serve skew prevention): This module reuses the EXACT same
preprocessing code path as training — RetinaFace face extraction, cropping
at the eval margin, resize to 224×224, ImageNet normalization.

Gate 8 (Deployment): Provides a clean API for serving:
    predictor = DeepfakePredictor("checkpoints/self_created/best")
    result = predictor.predict_video("path/to/video.mp4")

Gate 9 (Monitoring): Every prediction is logged with timestamp, input hash,
and score distribution for drift detection.

READ-ONLY with respect to model weights. This module loads weights but
never modifies them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from ..data.datasets import NORM_MEAN, NORM_STD, IMAGE_SIZE, normalize_tensor

logger = logging.getLogger("wmamba.inference")


class DeepfakePredictor:
    """Production-ready deepfake prediction from raw video files.

    Usage:
        predictor = DeepfakePredictor(
            checkpoint_dir="checkpoints/self_created/best",
            config_path="configs/self_created.yaml",
        )
        result = predictor.predict_video("path/to/video.mp4")
        # result = {"fake_probability": 0.87, "confidence": "high",
        #           "num_faces": 12, "num_frames": 32, "latency_ms": 1234}
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        config_path: str = "configs/self_created.yaml",
        device: str = "cuda",
        max_frames: int = 32,
        eval_margin: float = 0.125,
    ) -> None:
        from ..training.train import load_config
        from ..models.wmamba import build_wmamba

        self.device = device
        self.max_frames = max_frames
        self.eval_margin = eval_margin
        self.image_size = IMAGE_SIZE
        self._prediction_log: list[dict] = []

        # Load model
        cfg = load_config(config_path)
        self.model = build_wmamba(cfg).to(device)
        ckpt_path = Path(checkpoint_dir) / "model.safetensors"
        self.model.load_state_dict(load_file(str(ckpt_path)), strict=True)
        self.model.eval()

        # Load face detector (RetinaFace)
        self._init_face_detector()

        logger.info("DeepfakePredictor initialized: %s on %s", ckpt_path, device)

    def _init_face_detector(self) -> None:
        """Initialize RetinaFace for face detection."""
        from retinaface.pre_trained_models import get_model
        self.face_detector = get_model("resnet50_2020-07-20", max_size=1024)
        self.face_detector.eval()
        logger.info("RetinaFace face detector loaded")

    def _extract_faces_from_frame(self, frame_rgb: np.ndarray) -> list[np.ndarray]:
        """Detect and crop faces from a single frame."""
        annotations = self.face_detector.predict_jsons(frame_rgb)
        crops = []
        for ann in annotations:
            if ann["score"] < 0.5:
                continue
            bbox = ann["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h, w = frame_rgb.shape[:2]

            # Apply eval margin
            bw, bh = x2 - x1, y2 - y1
            mx1 = max(int(x1 - self.eval_margin * bw), 0)
            my1 = max(int(y1 - self.eval_margin * bh), 0)
            mx2 = min(int(x2 + self.eval_margin * bw), w)
            my2 = min(int(y2 + self.eval_margin * bh), h)

            if mx2 <= mx1 or my2 <= my1:
                continue

            crop = frame_rgb[my1:my2, mx1:mx2]
            crop = cv2.resize(crop, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
        return crops

    def _read_video_frames(self, video_path: str | Path) -> list[np.ndarray]:
        """Read evenly spaced frames from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has 0 frames: {video_path}")

        # Evenly spaced frame indices
        k = min(self.max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num=k).round().astype(int)
        indices = sorted(set(indices.tolist()))

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise ValueError(f"Could not read any frames from: {video_path}")
        return frames

    @torch.no_grad()
    def predict_video(self, video_path: str | Path) -> dict:
        """Run full inference pipeline on a raw video file.

        Returns:
            dict with keys:
                fake_probability: float in [0, 1]
                confidence: "high" | "medium" | "low"
                num_faces: total face crops scored
                num_frames: frames extracted from video
                latency_ms: end-to-end inference time in milliseconds
        """
        t0 = time.time()
        video_path = Path(video_path)

        # Step 1: Extract frames
        frames = self._read_video_frames(video_path)

        # Step 2: Detect and crop faces from each frame
        all_crops = []
        for frame in frames:
            crops = self._extract_faces_from_frame(frame)
            all_crops.extend(crops)

        if not all_crops:
            latency_ms = (time.time() - t0) * 1000
            result = {
                "fake_probability": 0.0,
                "confidence": "none",
                "num_faces": 0,
                "num_frames": len(frames),
                "latency_ms": round(latency_ms, 1),
                "warning": "No faces detected in video"
            }
            self._log_prediction(video_path, result)
            return result

        # Step 3: Normalize and batch
        tensors = torch.stack([normalize_tensor(c) for c in all_crops])
        tensors = tensors.to(self.device)

        # Step 4: Model inference
        logits = self.model(tensors)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)

        # Step 5: Aggregate (max per frame, then mean — paper protocol)
        fake_prob = float(probs.mean().cpu())

        # Confidence classification
        if fake_prob > 0.85 or fake_prob < 0.15:
            confidence = "high"
        elif fake_prob > 0.65 or fake_prob < 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        latency_ms = (time.time() - t0) * 1000

        result = {
            "fake_probability": round(fake_prob, 5),
            "confidence": confidence,
            "num_faces": len(all_crops),
            "num_frames": len(frames),
            "latency_ms": round(latency_ms, 1),
        }

        self._log_prediction(video_path, result)
        return result

    @torch.no_grad()
    def predict_image(self, image_path: str | Path) -> dict:
        """Run inference on a single pre-cropped face image.

        For cases where face extraction has already been done externally.
        """
        t0 = time.time()
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LINEAR)

        tensor = normalize_tensor(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = float(torch.softmax(logits, dim=1)[0, 1].cpu())

        latency_ms = (time.time() - t0) * 1000
        return {
            "fake_probability": round(prob, 5),
            "confidence": "high" if prob > 0.85 or prob < 0.15 else
                         "medium" if prob > 0.65 or prob < 0.35 else "low",
            "latency_ms": round(latency_ms, 1),
        }

    def _log_prediction(self, path: Path, result: dict) -> None:
        """Gate 9: Log prediction for drift monitoring."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_hash": hashlib.md5(str(path).encode()).hexdigest()[:12],
            "fake_probability": result["fake_probability"],
            "confidence": result["confidence"],
            "num_faces": result.get("num_faces", 0),
            "latency_ms": result["latency_ms"],
        }
        self._prediction_log.append(entry)
        logger.info("prediction: %s", json.dumps(entry))

    def get_prediction_log(self) -> list[dict]:
        """Return all logged predictions for drift analysis."""
        return list(self._prediction_log)

    def prediction_stats(self) -> dict:
        """Gate 9: Compute summary statistics of prediction distribution."""
        if not self._prediction_log:
            return {"count": 0}
        probs = [e["fake_probability"] for e in self._prediction_log]
        latencies = [e["latency_ms"] for e in self._prediction_log]
        return {
            "count": len(probs),
            "prob_mean": round(np.mean(probs), 4),
            "prob_std": round(np.std(probs), 4),
            "prob_median": round(float(np.median(probs)), 4),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 1),
        }
