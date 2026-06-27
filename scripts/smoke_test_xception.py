import sys, os
import yaml
import torch
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFB_ROOT = os.path.join(PROJECT_ROOT, "src", "external", "deepfakebench")
sys.path.insert(0, DFB_ROOT)

from detectors import DETECTOR
import detectors  # noqa: triggers registration of all 11 classes

config_path = os.path.join(DFB_ROOT, "config", "detector", "xception.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Override the broken relative path with our real local file
config["pretrained"] = os.path.join(DFB_ROOT, "pretrained", "xception-b5690688.pth")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_class = DETECTOR[config["model_name"]]
model = model_class(config).to(device)
model.eval()

ckpt_path = os.path.join(PROJECT_ROOT, "weights", "xception_best.pth")
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", len(missing), missing[:3])
print("Unexpected keys:", len(unexpected), unexpected[:3])

res = config["resolution"]
mean, std = config["mean"], config["std"]
dummy = torch.rand(1, 3, res, res).to(device)
dummy = T.Normalize(mean=mean, std=std)(dummy)

with torch.no_grad():
    pred_dict = model({"image": dummy}, inference=True)

print("Output keys:", list(pred_dict.keys()))
print("Fake probability (on random noise, not meaningful):", pred_dict["prob"].item())
print("\nSMOKE TEST PASSED")
