import warnings
warnings.filterwarnings("ignore")

import os, sys
import yaml
import torch
import gradio as gr
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_crop import FaceCropper, align_face
from src.utils.ensemble import aggregate
from src.detectors.dfb_detector import DFBDetector
from src.detectors.cnndetection_detector import CNNDetectionDetector
from src.detectors.univfd_detector import UnivFDDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading models once at startup (~30-60s)...")

with open(os.path.join(PROJECT_ROOT, "configs", "models.yaml")) as f:
    CFG = yaml.safe_load(f)

CROPPER = FaceCropper(device=DEVICE)

FACE_DETECTORS = []
GLOBAL_DETECTORS = []
for d in CFG["detectors"]:
    if d["type"] == "dfb":
        det = DFBDetector(d["name"], d["yaml"], d["checkpoint"], device=DEVICE)
        det.load()
        FACE_DETECTORS.append(det)
        print(f"  loaded (face-specific): {d['name']}")
    elif d["type"] == "standalone":
        if d["name"] == "cnndetection":
            det = CNNDetectionDetector(device=DEVICE)
        elif d["name"] == "univfd":
            det = UnivFDDetector(device=DEVICE)
        det.load()
        GLOBAL_DETECTORS.append(det)
        print(f"  loaded (whole-image generalist): {d['name']}")

print(f"All {len(FACE_DETECTORS) + len(GLOBAL_DETECTORS)} detectors loaded. Ready for uploads.")


def predict(image: Image.Image):
    if image is None:
        return "Please upload an image.", [], []

    global_results = [det.predict(image) for det in GLOBAL_DETECTORS]

    faces = CROPPER.detect(image)
    if not faces:
        summary_md = "**No face detected** — showing whole-image generalist results only:\n\n"
        table_rows = []
        for r in global_results:
            summary_md += f"- **{r['detector']}**: {r['label']} (fake_conf={r['fake_confidence']:.3f})\n"
            table_rows.append(["-", r["detector"], r["label"], round(r["fake_confidence"], 4)])
        return summary_md, [], table_rows

    summary_md = ""
    crops_gallery = []
    table_rows = []

    for r in global_results:
        table_rows.append(["all", r["detector"], r["label"], round(r["fake_confidence"], 4)])

    for i, face in enumerate(faces):
        per_detector = [det.predict(image, face["landmarks"]) for det in FACE_DETECTORS]
        for r in per_detector:
            table_rows.append([i + 1, r["detector"], r["label"], round(r["fake_confidence"], 4)])

        combined = per_detector + global_results
        result = aggregate(combined, threshold=CFG["ensemble"]["fake_threshold"])
        crops_gallery.append(align_face(image, face["landmarks"], res=256))

        icon = "🟥" if result["final_label"] == "fake" else "🟩"
        summary_md += (
            f"### Face {i+1} {icon} {result['final_label'].upper()}\n"
            f"- **Confidence (fake):** {result['final_confidence']*100:.1f}%\n"
            f"- **Detector agreement:** {result['agreement']}\n"
            f"- **Range across detectors:** {result['min_confidence']*100:.1f}% – {result['max_confidence']*100:.1f}%\n\n"
        )

    return summary_md, crops_gallery, table_rows


with gr.Blocks(title="Deepfake Image Detector") as demo:
    gr.Markdown(
        "# Deepfake Image Detector\n"
        f"Upload an image. **{len(FACE_DETECTORS)} face-specific detectors** analyze any detected "
        f"face(s), and **{len(GLOBAL_DETECTORS)} whole-image generalist detectors** (GAN/diffusion-focused) "
        "analyze the full photo. All results combine into one verdict per face."
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload image")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            output_summary = gr.Markdown(label="Result")
            output_gallery = gr.Gallery(label="Detected face crop(s)", columns=4, height=200)

    output_table = gr.Dataframe(
        headers=["Face #", "Detector", "Label", "Fake confidence"],
        label="Per-detector breakdown",
    )

    submit_btn.click(fn=predict, inputs=input_image, outputs=[output_summary, output_gallery, output_table])

if __name__ == "__main__":
    demo.launch(share=True)
