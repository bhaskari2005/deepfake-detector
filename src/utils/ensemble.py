"""
Combines per-detector predictions into one final verdict + summary.
"""
from typing import List, Dict


def aggregate(detector_results: List[Dict], threshold: float = 0.5) -> Dict:
    if not detector_results:
        return {
            "final_label": "unknown",
            "final_confidence": None,
            "num_detectors": 0,
            "agreement": None,
            "breakdown": [],
        }

    confidences = [r["fake_confidence"] for r in detector_results]
    mean_conf = sum(confidences) / len(confidences)
    n_fake_votes = sum(1 for r in detector_results if r["label"] == "fake")
    n_total = len(detector_results)

    final_label = "fake" if mean_conf >= threshold else "real"

    return {
        "final_label": final_label,
        "final_confidence": mean_conf,
        "num_detectors": n_total,
        "agreement": f"{n_fake_votes}/{n_total} detectors said FAKE",
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "breakdown": sorted(detector_results, key=lambda r: r["fake_confidence"], reverse=True),
    }
