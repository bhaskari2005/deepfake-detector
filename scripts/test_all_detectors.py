"""
Batch smoke test for all 11 registered DeepfakeBench detectors.
Tries each one independently — a failure in one does NOT stop the others.
Prints one consolidated report at the end.
"""

import sys, os, traceback
import yaml
import torch
import torchvision.transforms as T

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFB_ROOT = os.path.join(PROJECT_ROOT, "src", "external", "deepfakebench")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
PRETRAINED_DIR = os.path.join(DFB_ROOT, "pretrained")

sys.path.insert(0, DFB_ROOT)

XCEPTION_BACKBONE = os.path.join(PRETRAINED_DIR, "xception-b5690688.pth")
EFFNB4_BACKBONE = os.path.join(PRETRAINED_DIR, "efficientnet-b4-6ed6700e.pth")

DETECTORS = [
    ("xception.yaml", "xception_best.pth"),
    ("efficientnetb4.yaml", "effnb4_best.pth"),
    ("ucf.yaml", "ucf_best.pth"),
    ("spsl.yaml", "spsl_best.pth"),
    ("f3net.yaml", "f3net_best.pth"),
    ("srm.yaml", "srm_best.pth"),
    ("ffd.yaml", "ffd_best.pth"),
    ("core.yaml", "core_best.pth"),
    ("recce.yaml", "recce_best.pth"),
    ("capsule_net.yaml", "capsule_best.pth"),
    ("meso4Inception.yaml", "meso4Incep_best.pth"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

results = []

for yaml_name, ckpt_name in DETECTORS:
    tag = yaml_name.replace(".yaml", "")
    print(f"--- Testing: {tag} " + "-" * (40 - len(tag)))
    try:
        from detectors import DETECTOR

        config_path = os.path.join(DFB_ROOT, "config", "detector", yaml_name)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pretrained_path = config.get("pretrained", False)
        if isinstance(pretrained_path, str):
            if "xception" in pretrained_path:
                config["pretrained"] = XCEPTION_BACKBONE
            elif "efficientnet" in pretrained_path:
                config["pretrained"] = EFFNB4_BACKBONE
            else:
                print(f"  WARNING: unrecognized pretrained path '{pretrained_path}', leaving as-is")

        model_name = config.get("model_name", tag)
        model_class = DETECTOR[model_name]
        model = model_class(config).to(device)
        model.eval()

        ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        res = config.get("resolution", 256)
        mean = config.get("mean", [0.5, 0.5, 0.5])
        std = config.get("std", [0.5, 0.5, 0.5])
        dummy = torch.rand(1, 3, res, res).to(device)
        dummy = T.Normalize(mean=mean, std=std)(dummy)

        with torch.no_grad():
            pred_dict = model({"image": dummy, "label": torch.zeros(1, dtype=torch.long).to(device)}, inference=True)

        prob = pred_dict.get("prob")
        prob_val = prob.item() if prob is not None else None

        print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        print(f"  Output keys: {list(pred_dict.keys())}")
        print(f"  Dummy fake-prob: {prob_val}")
        print(f"  PASSED")

        results.append((tag, "PASS", f"missing={len(missing)} unexpected={len(unexpected)}"))

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        results.append((tag, "FAIL", f"{type(e).__name__}: {e}"))

    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
for tag, status, detail in results:
    print(f"{status:5s} | {tag:20s} | {detail}")

n_pass = sum(1 for _, s, _ in results if s == "PASS")
print(f"\n{n_pass}/{len(results)} detectors passed.")
