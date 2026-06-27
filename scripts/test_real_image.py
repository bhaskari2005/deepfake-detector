import sys, os
import yaml
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.detectors.dfb_detector import DFBDetector

config_path = os.path.join(PROJECT_ROOT, "configs", "models.yaml")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

image_path = os.path.join(PROJECT_ROOT, "data", "samples", "test_face.jpg")
image = Image.open(image_path)
print(f"Testing on: {image_path} ({image.size})\n")

results = []
for det_cfg in cfg["detectors"]:
    name = det_cfg["name"]
    print(f"--- {name} ---")
    try:
        detector = DFBDetector(name, det_cfg["yaml"], det_cfg["checkpoint"])
        result = detector.predict(image)
        print(f"  {result}")
        results.append(result)
        del detector
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()

print("=" * 50)
print(f"{len(results)}/{len(cfg['detectors'])} detectors produced a result")
if results:
    avg = sum(r["fake_confidence"] for r in results) / len(results)
    print(f"Average fake-confidence across all detectors: {avg:.4f}")
