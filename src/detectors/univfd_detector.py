"""
Wrapper for UnivFD (Ojha et al., CVPR 2023) - "Towards Universal Fake Image
Detectors that Generalize Across Generative Models". Frozen CLIP ViT-L/14
visual encoder + a single trained linear probe. Built specifically to
generalize across GAN/diffusion generators it never saw during training -
not a face-swap detector, so it should see the original unaligned image.
"""
import os
import sys
import torch
from PIL import Image

from .base import BaseDeepfakeDetector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIVFD_ROOT = os.path.join(PROJECT_ROOT, "src", "external", "univfd")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "univfd_fc_weights.pth")

if UNIVFD_ROOT not in sys.path:
    sys.path.insert(0, UNIVFD_ROOT)


class UnivFDDetector(BaseDeepfakeDetector):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.name = "univfd"
        self.clip_preprocess = None

    def load(self):
        from models import get_model  # vendored package, see UNIVFD_ROOT

        model = get_model("CLIP:ViT-L/14")  # downloads CLIP's own weights on first use (~890MB)
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.fc.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.clip_preprocess = model.preprocess  # CLIP's own resize/crop/normalize

    def preprocess(self, image: Image.Image):
        image = image.convert("RGB")
        tensor = self.clip_preprocess(image)
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
