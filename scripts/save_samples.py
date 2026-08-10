import sys
import torch
import torchvision
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.train import load_config
from src.data.datasets import SupervisedTrainDataset

def main():
    print("Loading config and dataset...")
    cfg = load_config("configs/lavdf.yaml")
    dataset = SupervisedTrainDataset(Path(cfg.paths.processed) / "lavdf", dataset_name="lavdf", split="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    videos, labels = batch
    
    os.makedirs("samples", exist_ok=True)
    out_path = "samples/sanity_check_faces.jpg"
    
    # Normalize=True automatically scales the tensor min/max to 0-255 for the image
    torchvision.utils.save_image(videos, out_path, normalize=True, nrow=4)
    print(f"Saved {videos.shape[0]} samples to {out_path}")
    print(f"Labels for these images: {labels.tolist()} (0=Fake, 1=Real)")

if __name__ == "__main__":
    main()
