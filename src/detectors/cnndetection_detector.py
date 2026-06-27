"""
Wrapper for CNNDetection (Wang et al. 2020) - "CNN-generated images are
surprisingly easy to spot... for now". ResNet50 trained on ProGAN outputs;
known to generalize well across many GAN architectures (not face-swap
deepfakes - it targets a different forgery paradigm entirely).
"""
import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .base import BaseDeepfakeDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CNNDET_ROOT = os.path.join(PROJECT_ROOT, "src", "external", "cnndetection")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "cnndetection_blur_jpg_prob0.5.pth")

if CNNDET_ROOT not in sys.path:
    sys.path.insert(0, CNNDET_ROOT)


class CNNDetectionDetector(BaseDeepfakeDetector):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.name = "cnndetection"

    def load(self):
        from resnet import resnet50  # vendored file, see CNNDET_ROOT

        model = resnet50(num_classes=1)
        state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
        model.load_state_dict(state_dict["model"])
        model.to(self.device)
        model.eval()
        self.model = model

    def preprocess(self, image: Image.Image):
        # matches their demo.py exactly: no cropping, ImageNet normalization
        image = image.convert("RGB")
        tensor = T.ToTensor()(image)
        tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> dict:
        if self.model is None:
            self.load()

        tensor = self.preprocess(image)
        with torch.no_grad():
            logit = self.model(tensor)
            fake_conf = torch.sigmoid(logit).item()

        return {
            "detector": self.name,
            "label": "fake" if fake_conf >= 0.5 else "real",
            "fake_confidence": fake_conf,
            "raw_output": None,
        }
