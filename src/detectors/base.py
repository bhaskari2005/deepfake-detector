from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image


class BaseDeepfakeDetector(ABC):
    name: str = "base"

    def __init__(self, weights_path: str = None, device: str = "cuda"):
        self.weights_path = weights_path
        self.device = device
        self.model = None

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image: Image.Image):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict:
        raise NotImplementedError
