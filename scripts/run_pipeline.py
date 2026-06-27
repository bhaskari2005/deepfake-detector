import sys, os
import yaml
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_crop import FaceCropper
from src.utils.ensemble import aggregate
from src.detectors.dfb_detector import DFBDetector
from src.detectors.cnndetection_detector import CNNDetectionDetector
from src.detectors.univfd_detector import UnivFDDetector


def build_detectors(cfg, device):
    face_detectors = []
    global_detectors = []
    for d in cfg["detectors"]:
        if d["type"] == "dfb":
            face_detectors.append(DFBDetector(d["name"], d["yaml"], d["checkpoint"], device=device))
        elif d["type"] == "standalone":
            if d["name"] == "cnndetection":
                global_detectors.append(CNNDetectionDetector(device=device))
            elif d["name"] == "univfd":
                global_detectors.append(UnivFDDetector(device=device))
    return face_detectors, global_detectors


def run_pipeline(image_path, config_path=None):
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "models.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    image = Image.open(image_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cropper = FaceCropper(device=device)
    faces = cropper.detect(image)

    face_detectors, global_detectors = build_detectors(cfg, device)

    print(f"\n=== Whole-image generalist detectors (GAN/diffusion-focused) ===")
    global_results = []
    for det in global_detectors:
        r = det.predict(image)
        global_results.append(r)
        print(f"  {r['detector']:18s} -> {r['label']:5s} (fake_conf={r['fake_confidence']:.3f})")

    if not faces:
        print("\nNo face detected — skipping face-specific detectors.")
        return {"faces": [], "global": global_results}

    all_results = []
    for i, face in enumerate(faces):
        print(f"\n=== Face {i} (detector_confidence={face['confidence']:.3f}, box={face['box']}) ===")
        per_detector = []
        for det in face_detectors:
            r = det.predict(image, face["landmarks"])
            per_detector.append(r)
            print(f"  {r['detector']:18s} -> {r['label']:5s} (fake_conf={r['fake_confidence']:.3f})")

        combined = per_detector + global_results
        summary = aggregate(combined, threshold=cfg["ensemble"]["fake_threshold"])
        print(f"\n  FINAL VERDICT (face-specific + global): {summary['final_label'].upper()} "
              f"(confidence={summary['final_confidence']:.3f}, {summary['agreement']})")
        all_results.append({"face_index": i, "box": face["box"], "summary": summary})

    torch.cuda.empty_cache()
    return {"faces": all_results, "global": global_results}


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PROJECT_ROOT, "data", "samples", "test_face.jpg")
    run_pipeline(image_path)
