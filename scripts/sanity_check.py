import sys
import math
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.wmamba import build_wmamba
from src.data.datasets import SupervisedTrainDataset

def main():
    print("Loading config...")
    from src.training.train import load_config
    cfg = load_config("configs/lavdf.yaml")
    
    print("Building model...")
    model = build_wmamba(cfg).cuda()
    model.train()

    print("Loading dataset...")
    dataset = SupervisedTrainDataset(Path(cfg.paths.processed) / "lavdf", dataset_name="lavdf", split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    videos, labels = batch
    videos = videos.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        logits = model(videos)
        loss = F.cross_entropy(logits, labels)
        
    print("-" * 50)
    print(f"[Sanity] Input shape: {videos.shape}, Labels: {labels}")
    print(f"[Sanity] Loss at init: {loss.item():.4f}")
    print(f"[Sanity] Expected init loss: {-math.log(1/cfg.model.num_classes):.4f}")
    print("-" * 50)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    print("[Sanity] Overfitting single batch (50 steps)...")
    for i in range(50):
        optimizer.zero_grad()
        logits = model(videos)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            print(f"Step {i}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

    logits = model(videos)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    print(f"Final Step: Loss {loss.item():.4f}, Acc {acc.item():.4f}")

if __name__ == "__main__":
    main()
