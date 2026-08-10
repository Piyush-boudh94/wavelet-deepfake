import sys
import torch
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.models.wmamba import build_wmamba
from src.eval.protocols import evaluate_within_dataset

def main():
    print("Loading config...")
    cfg = load_config("configs/self_created.yaml")
    device = "cuda"
    
    print("Building model...")
    model = build_wmamba(cfg).to(device)
    
    ckpt_path = "checkpoints/self_created/best/model.safetensors"
    print(f"Loading weights from {ckpt_path}...")
    weights = load_file(ckpt_path)
    model.load_state_dict(weights, strict=True)
    model.eval()

    print("Evaluating on self_created test split...")
    results = evaluate_within_dataset(
        model=model,
        processed_root=Path(cfg.paths.processed) / "self_created",
        dataset_name="self_created",
        batch_size=64,
        device=device,
        num_workers=8,
        splits=("test",),
        frames_per_video=32
    )
    
    print("-" * 50)
    print("FINAL PHASE 2 TEST RESULTS")
    print(f"Test AUC: {results['test']:.5f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
