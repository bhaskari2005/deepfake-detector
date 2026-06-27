import sys, os
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_crop import FaceCropper

image_path = os.path.join(PROJECT_ROOT, "data", "samples", "test_face.jpg")
image = Image.open(image_path)
print(f"Original image size: {image.size}")

cropper = FaceCropper(device="cuda", margin_scale=1.3)
faces = cropper.detect_and_crop(image)

print(f"Faces detected: {len(faces)}")
for i, f in enumerate(faces):
    out_path = os.path.join(PROJECT_ROOT, "data", "samples", f"cropped_face_{i}.jpg")
    f["crop"].save(out_path)
    print(f"  Face {i}: box={f['box']} confidence={f['confidence']:.3f} "
          f"crop_size={f['crop'].size} saved_to={out_path}")
