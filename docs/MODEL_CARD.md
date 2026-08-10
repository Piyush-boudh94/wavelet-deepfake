# Model Card: WMamba Deepfake Detection

*Following [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993) — "Model Cards for Model Reporting"*

## Model Details

| Field | Value |
|---|---|
| **Model Name** | WMamba (Wavelet-based Mamba for Face Forgery Detection) |
| **Architecture** | VMamba-S backbone + HWFEB (Haar Wavelet Feature Enhancement Block) + DCConv (Direction-aware Convolution) |
| **Parameters** | ~50M |
| **Training Framework** | PyTorch 2.10.0 + mamba_ssm 2.3.2 + causal_conv1d 1.6.2 |
| **Precision** | BF16 autocast (training); FP32 weights |
| **Input** | 224×224 RGB face crops (RetinaFace extraction) |
| **Output** | Binary classification: Real (0) vs Fake (1), softmax probability |
| **License** | Research use; VMamba backbone is MIT licensed |
| **Paper** | Peng et al., "WMamba: Wavelet-based Mamba for Face Forgery Detection", ACM MM 2025 (arXiv:2501.09617) |
| **Date** | 2026-08-07 |

## Intended Use

### Primary Use
Detection of face-swapped deepfake videos. The model accepts individual face crops extracted from video frames and outputs a probability of the face being manipulated.

### Out-of-Scope Uses
- **Audio-only deepfakes:** The model operates on visual data only. It cannot detect audio manipulation.
- **Non-face content:** The model expects cropped face regions. Feeding it full scenes, objects, or non-face imagery will produce meaningless outputs.
- **Real-time video surveillance:** Inference latency has not been optimized for real-time streaming applications.
- **Legal evidence:** Model predictions should not be used as sole evidence in legal proceedings without human expert review.

## Training Data

### Phase 1 — LAV-DF (Pre-training)
- **Source:** LAV-DF dataset (Large-scale Audio-Visual Deepfake)
- **Videos:** 30,000 subsampled from 136,304 (train: 20,000, val: 4,000, test: 6,000)
- **Label Rule:** `modify_video` flag (visual forgery only; audio-only fakes excluded)
- **Class Balance:** ~49% fake per split
- **Preprocessing:** RetinaFace face extraction → 16 frames/video (train), 32 frames/video (eval)

### Phase 2 — self_created (Fine-tuning)
- **Source:** Curated dataset combining clips from DFDC, FF++, and LAV-DF
- **Videos:** 3,288 total (train: 2,300, val: 494, test: 494)
- **Preprocessing:** Same pipeline as Phase 1
- **Transfer:** Phase 1 best weights (AUC 0.99623) used as initialization

## Evaluation Results

| Dataset | Split | Metric | Score |
|---|---|---|---|
| LAV-DF | val | Video AUC | 0.99623 |
| self_created | test | Video AUC | **0.88321** |

### Evaluation Protocol
- Frame-level scores averaged per video, then ROC-AUC computed over all videos in the split
- Multi-face frames: max fakeness score taken across faces before per-video averaging
- 32 frames per video, evenly spaced, at fixed 12.5% crop margin

## Known Limitations

1. **Domain Shift:** The model was fine-tuned on a specific mix of DFDC, FF++, and LAV-DF clips. Performance on unseen deepfake generation methods (e.g., newer diffusion-based face swaps) is unknown.
2. **Compression Sensitivity:** Error analysis revealed that heavily compressed videos (common in DFDC) produce false positives — the model may confuse compression artifacts with manipulation artifacts.
3. **No OOD Detection:** The model always produces a prediction. It has no mechanism to flag inputs that are unlike anything seen during training.
4. **Probability Calibration:** Raw softmax outputs have not been calibrated; the predicted probability may not reflect true likelihood of forgery.
5. **Single-GPU Training:** Trained on a single MIG slice (H200 1g.18gb, 16 GB). Not validated on multi-GPU or different hardware.

## Deviations from Reference Paper

1. BF16 autocast (paper implies FP32)
2. WFEM internal channel counts inferred (paper underspecified)
3. Angle-predictor bias −6 initialization (paper silent)
4. Checkpoint every 2 epochs (cluster stability)
5. Triton cross-scan disabled (no compiler in pod)
6. drop_path 0.3 (VMamba-S official recipe)
7. Datasets are DFDC + LAV-DF only (not paper's five)
8. SBI disabled; supervised training on real labels
9. LAV-DF label = `modify_video`, not `n_fakes`
10. DFDC split is custom-built (identity-grouped)

Full details in `AUDIT.md`.

## Ethical Considerations

- **Dual Use:** Deepfake detection technology can be used both defensively (identifying misinformation) and to study weaknesses of detection systems. This model is intended for defensive use.
- **Demographic Bias:** Face detection models (RetinaFace) have documented biases across skin tones and lighting conditions. This may cause uneven detection performance across demographic groups. No demographic audit was performed due to lack of metadata.
- **False Positives:** Real videos incorrectly flagged as fake can cause reputational harm. The model should never be the sole decision-maker.

## Infrastructure

- **Checkpointing:** Atomic writes on NFS with immediate reload + spot-checksum verification
- **Reproducibility:** All 4 RNG streams (python, numpy, torch, cuda) seeded and saved in checkpoints
- **Provenance:** Each checkpoint includes `meta.json` with full resolved config, git commit hash, and lockfile SHA-256
