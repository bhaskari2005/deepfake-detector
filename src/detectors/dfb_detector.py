"""
Generic wrapper around any DeepfakeBench detector.
One class handles all 11 — just pass different yaml/checkpoint names.
"""
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .base import BaseDeepfakeDetector
from src.utils.face_crop import align_face

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DFB_ROOT = os.path.join(PROJECT_ROOT, "src", "external", "deepfakebench")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
PRETRAINED_DIR = os.path.join(DFB_ROOT, "pretrained")

XCEPTION_BACKBONE = os.path.join(PRETRAINED_DIR, "xception-b5690688.pth")
EFFNB4_BACKBONE = os.path.join(PRETRAINED_DIR, "efficientnet-b4-6ed6700e.pth")

# sys.path handled inside load() instead, see below


class DFBDetector(BaseDeepfakeDetector):
    def __init__(self, name, yaml_name, ckpt_name, device="cuda"):
        super().__init__(device=device)
        self.name = name
        self.yaml_name = yaml_name
        self.ckpt_name = ckpt_name
        self.config = None

    def load(self):
        if DFB_ROOT in sys.path:
            sys.path.remove(DFB_ROOT)
        sys.path.insert(0, DFB_ROOT)
        from detectors import DETECTOR  # local import: registers all 11 on first call

        config_path = os.path.join(DFB_ROOT, "config", "detector", self.yaml_name)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pretrained_path = config.get("pretrained", False)
        if isinstance(pretrained_path, str):
            if "xception" in pretrained_path:
                config["pretrained"] = XCEPTION_BACKBONE
            elif "efficientnet" in pretrained_path:
                config["pretrained"] = EFFNB4_BACKBONE

        model_name = config.get("model_name", self.name)
        model_class = DETECTOR[model_name]
        model = model_class(config).to(self.device)
        model.eval()

        ckpt_path = os.path.join(WEIGHTS_DIR, self.ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)

        self.model = model
        self.config = config

    def preprocess(self, image: Image.Image, landmarks):
        res = self.config.get("resolution", 256)
        mean = self.config.get("mean", [0.5, 0.5, 0.5])
        std = self.config.get("std", [0.5, 0.5, 0.5])

        aligned = align_face(image, landmarks, res=res, scale=1.3)
        tensor = T.ToTensor()(aligned)
        tensor = T.Normalize(mean=mean, std=std)(tensor)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, landmarks) -> dict:
        if self.model is None:
            self.load()

        tensor = self.preprocess(image, landmarks)
        dummy_label = torch.zeros(1, dtype=torch.long).to(self.device)

        with torch.no_grad():
            pred_dict = self.model({"image": tensor, "label": dummy_label}, inference=True)

        if "prob" in pred_dict and pred_dict["prob"] is not None:
            fake_conf = pred_dict["prob"].item()
        else:
            # fallback: derive from raw class logits (e.g. UCF in inference mode)
            fake_conf = F.softmax(pred_dict["cls"], dim=1)[:, 1].item()

        return {
            "detector": self.name,
            "label": "fake" if fake_conf >= 0.5 else "real",
            "fake_confidence": fake_conf,
            "raw_output": None,
        }
