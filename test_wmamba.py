from omegaconf import OmegaConf
from src.models.wmamba import build_wmamba
from src.training.train import load_config
import sys

cfg = load_config("configs/phase4b.yaml")
print(f"Config architecture_mode: {cfg.model.get('architecture_mode', 'baseline')}")
model = build_wmamba(cfg)
print(f"Model type: {type(model)}")
if hasattr(model, "param_groups"):
    print("Model has param_groups")
else:
    print("Model DOES NOT have param_groups")
