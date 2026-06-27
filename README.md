# Deepfake Image Detection — Ensemble of Pretrained Detectors

Upload an image → detect any face(s) → run them through **11 face-specific
forgery detectors** + **2 whole-image GAN-detection generalists** → combine
all 13 outputs into one confidence score and per-detector summary per face.

Built and tested on PARAM Rudra (A100 80GB, CUDA 12.4, PyTorch 2.2.1+cu121).

---

## How it works
## The 13 detectors

### Face-specific (11) — trained on FaceForensics++ (real YouTube videos + 5 face-swap/reenactment methods: DeepFakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter)

Source: [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) (NeurIPS 2023)

| Model | Mechanism | Best at |
|---|---|---|
| Xception | Spatial CNN | Classic FF++-style face-swaps |
| EfficientNet-B4 | Spatial CNN, different backbone | Same as Xception |
| UCF | Disentangles forgery-specific vs. common features | Best overall in-domain performer |
| SPSL | Frequency phase + spatial | Best cross-dataset generalization (within face-swaps) |
| F3-Net | Frequency-domain decomposition | Compression-related artifacts |
| SRM | Noise-residual filters | Noise-pattern cues spatial models miss |
| FFD | Predicts blending mask + label | Localizing the swap boundary |
| CORE | Consistency regularization | Robustness to blur/compression |
| RECCE | Reconstruction-based anomaly detection | Slightly novel fakes (models "real," not one fake type) |
| Capsule-Net | Capsule network (non-CNN) | Ensemble diversity (different error patterns) |
| MesoInception | Lightweight mesoscale CNN | Fast; sometimes catches generic texture issues others miss |

### Whole-image generalists (2) — trained on ProGAN-generated images

| Model | Mechanism | Best at | Source |
|---|---|---|---|
| CNNDetection | ResNet50 + heavy blur/JPEG augmentation | Generalizing across many GAN architectures it never saw | [PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection) (CVPR 2020) |
| UnivFD | Frozen CLIP ViT-L/14 + linear probe | Broadest cross-generator reach (semantic features, not GAN-specific artifacts) | [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) (CVPR 2023) |

## Setup

```bash
conda create -n dfdetect python=3.10 -y
conda activate dfdetect
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install efficientnet_pytorch einops imageio tensorboard ftfy regex
```

**Pretrained weights are not committed to this repo** (several exceed GitHub's 100MB limit; total set is >1GB). They must be downloaded separately:
- 11 DeepfakeBench checkpoints: `github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1`
- CNNDetection: `huggingface.co/nebula/testmodel` (blur_jpg_prob0.5.pth)
- UnivFD: committed directly in their repo (`pretrained_weights/fc_weights.pth`)
- Two ImageNet backbone placeholders (Xception, EfficientNet-B4) are also required — see `src/external/deepfakebench/pretrained/`

## Usage

Command line:
```bash
python scripts/run_pipeline.py path/to/image.jpg
```

Web UI:
```bash
python src/api/app.py
```
Then open the printed local URL (use an SSH tunnel if running on a remote cluster).

## Known limitations

- **Strong on**: FaceForensics++-style face-swap/reenactment forgeries (11 of 13 detectors) and classic GAN architectures (2 of 13).
- **Weak on**: modern diffusion-generated/edited content (no detector here targets this specifically — DIRE was considered but requires running a full diffusion model's reverse process per image, a much heavier addition); generators explicitly designed to evade texture-based detection (e.g. StyleGAN3).
- **Ensemble logic**: current aggregation is a simple mean across all 13, which can dilute a strong individual signal from 1-2 specialists. A confidence-weighted or "any confident specialist flags it" rule would likely perform better in practice.
- This mirrors the field's broader open problem: no current detector (academic or commercial) generalizes reliably across *all* generation methods, especially as new ones emerge.

## Acknowledgments

This project integrates pretrained models and code from:
- Yan et al., *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection* (NeurIPS 2023)
- Wang et al., *CNN-generated images are surprisingly easy to spot... for now* (CVPR 2020)
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023)

All model weights and architectures are used under their respective original licenses, for academic/educational purposes.
