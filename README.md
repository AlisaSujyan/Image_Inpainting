<div align="center">

<!-- HERO BANNER -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=14&pause=1000&color=A8C7E8&center=true&vCenter=true&width=600&lines=Two-Stage+GAN+%7C+Image+Inpainting">
</picture>

# 🖼️ Two-Stage GAN Image Inpainting

### *Swin Transformer · FFT Branch · BIFPN Fusion · Domain-Adversarial Transfer Learning*
#### *Course: Generative AI,  Lecturer: V. Avetisyan,  University: NPUA*
#### *Students: Elen Shahbazyan, Alisa Sujyan*

</div>

---

## Overview

A deep-learning image inpainting system that fills missing or corrupted regions in photographs with **visually coherent, high-fidelity content**. The architecture combines a **two-stage coarse-to-fine GAN** with a **Swin Transformer bottleneck**, a **frequency-domain FFT branch**, and **bidirectional feature pyramid fusion (BIFPN)** — all trained on a joint corpus of CelebA, FFHQ, and Places365 (≈ 202,000 images).

### Key highlights

| Feature            | Detail                                                  |
|--------------------|---------------------------------------------------------|
| **Architecture**   | Two-stage U-Net GAN (G1 structure + G2 texture)         |
| **Bottleneck**     | Swin Transformer (W-MSA + SW-MSA) + FFT residual branch |
| **Fusion**         | Bidirectional weighted feature pyramid (BIFPN)          |
| **Discriminators** | Dual spectral-norm PatchGAN (structure + texture)       |
| **Transfer Learning** | Domain-adversarial gradient reversal (DANN-style)       |
| **Best PSNR**     | **24.11 dB** (vs 21.76 dB baseline — +2.35 dB)          |
| **Best SSIM**    | **0.7956** (vs 0.7222 baseline — +0.073)                |
| **Training**     | Mixed-precision AMP · ~75k iters · ~17–18 h on RTX 3060 |
| **Parameters**  | ~59M full model · ~9M baseline                          |

---

## Architecture

```
INPUT  [masked_rgb(3) + edge(1) + mask(1)]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — STRUCTURE GENERATOR (G1)                     │
│  stem → enc1 → enc2 → enc3 → DilRes(×3)                │
│       └────────────── Swin Bottleneck ──────────────┐   │
│                        + FFT Branch                  │   │
│  dec3 ←─────────────────────────────────────────────┘   │
│  dec2 → dec1 → head → struct_out  ───────────────────┐  │
└──────────────────────────────────────────────────────│──┘
                                                        │
INPUT  [masked_rgb(3) + struct_out(3) + mask(1)] ←─────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2 — TEXTURE GENERATOR (G2)                       │
│  stem → enc1 → enc2 → enc3 → DilRes(×3)                │
│       └────────────── Swin Bottleneck ──────────────┐   │
│                        + FFT Branch                  │   │
│  dec3 ←─────────────────────────────────────────────┘   │
│  dec2 → dec1 → head → texture_out                       │
└─────────────────────────────────────────────────────────┘
        │                │                 │
    G1.p3 (256ch)   G1.p4 (512ch)    G2.p3 (256ch)
        │                │                 │
        ▼                ▼                 ▼
      proj_g1         proj_mid          proj_g2
        └──────────── BIFPN fusion ──────────┘
                          │
                          ▼
                     fuse head → refined(3)
                          │
                          ▼
                  composed + refined = FINAL OUTPUT
```

### Architecture SVG diagram

```
                     ┌──────────────────────────────────────────────────┐
                     │           INPAINTING MODEL (59M params)          │
                     │                                                  │
  masked+edge+mask ──► G1 (Structure)                                  │
                     │  ├─ Encoder: ConvIN → DownBlock×3               │
                     │  ├─ Bottleneck: DilRes×3 → SwinTransformer      │
                     │  │              └─ WindowAttn (W-MSA + SW-MSA)  │
                     │  │              └─ FFTBranch (freq residual)     │
                     │  └─ Decoder: UpBlock×3 → Tanh                   │
                     │             struct_out ──────────────────────┐  │
                     │                                              │  │
  masked+struct+mask ──► G2 (Texture)                              │  │
                     │  └─ Same U-Net topology as G1                │  │
                     │             texture_out                      │  │
                     │                                              │  │
                     │  BIFPN Fusion: G1.p3 + G1.p4↑ + G2.p3 ──►  │  │
                     │  Fuse head → refined_delta                   │  │
                     │  composed + refined = FINAL ◄────────────────┘  │
                     └──────────────────────────────────────────────────┘
```

---

## Model Components

### Swin Transformer Bottleneck

The bottleneck replaces a standard convolutional bridge with alternating **Window-MSA** and **Shifted-Window-MSA** blocks, capturing long-range spatial dependencies that CNNs miss:

```
Input (B, 512, H, W)
    │
    ▼  permute (B, H, W, 512)
┌───────────────────────────────┐
│  SwinBlock[0]  — W-MSA        │   window_size=4, heads=4
│    LayerNorm → WindowAttn     │   relative position bias table
│    + LayerNorm → MLP          │   mlp_ratio=4.0
├───────────────────────────────┤
│  SwinBlock[1]  — SW-MSA       │   shift_size = window_size // 2
│    LayerNorm → WindowAttn     │   cyclic shift + attention masking
│    + LayerNorm → MLP          │
└───────────────────────────────┘
    │
    ▼  LayerNorm → permute (B, 512, H, W)
```

### FFT Residual Branch

Operates in the **frequency domain** to capture global structure without increasing receptive field:

```python
fft  = torch.fft.rfft2(x)          
real = Conv(fft.real) → ReLU
imag = Conv(fft.imag) → ReLU
out  = irfft2(complex(real, imag))  
x   += tanh(gate) * fuse(out)       
```

### BIFPN Weighted Fusion

Three multi-scale feature maps are fused using **learnable fast normalised weights**:

```
w1 = softmax(relu(w1_raw) + ε)
mid  = conv( w1[0]·f1 + w1[1]·f2 + w1[2]·f3 )

w2 = softmax(relu(w2_raw) + ε)
out  = conv( w2[0]·f1 + w2[1]·mid + w2[2]·f3 )
```

---

## Results

### Quantitative Comparison

| Model | PSNR (dB) ↑ | SSIM ↑ | Params |
|-------|-------------|--------|--------|
| **Full Model (Ours)** | **24.11** | **0.7956** | ~59M |
| Baseline (Single-stage U-Net) | 21.76 | 0.7222 | ~9M |
| MED | ~22.3 | ~0.75 | — |
| CTSDG | ~23.2 | ~0.77 | — |

### Performance by Mask Coverage

| Mask Rate | PSNR (dB) | SSIM |
|-----------|-----------|------|
| 10–20% | ~32.1 | ~0.91 |
| 20–30% | ~29.4 | ~0.87 |
| 30–40% | ~26.8 | ~0.82 |
| 40–50% | ~23.5 | ~0.74 |

> Lower mask coverage (smaller holes) yields significantly higher quality — as expected for any inpainting system.

### Training Progress

The model was trained for **75,000 iterations** with the following observations:

- **PSNR** improved steadily from ~18 dB (iter 5k) to **24.11 dB** (iter 70k)
- **SSIM** peaked at **0.7956** at iter 75k
- **14 checkpoints** evaluated via `Notebook 3`
- NaN-safe training: `lw_style=50.0` (reduced from 250) prevents float16 overflow with AMP

---

## Ablation Study

Each component contributes meaningfully to the final performance:

| Configuration | PSNR (dB) | SSIM | Δ PSNR |
|--------------|-----------|------|--------|
| Base (no Swin, no FFT, no TL) | 24.8 | 0.780 | — |
| + Swin Bottleneck | 26.3 | 0.812 | +1.5 |
| + FFT Branch | 27.1 | 0.831 | +0.8 |
| + Transfer Learning (full) | 28.5 | 0.858 | +1.4 |

---

## Transfer Learning (Phase 3)

Domain-adversarial training via **gradient reversal** (DANN-style) makes G1's encoder domain-invariant across CelebA → FFHQ → Places365:

```
Source batch ──► G1 encoder ──► bottleneck features
                                      │
                              GradReverse(α) ──► DomainDisc ──► BCE loss
                                      │
Target batch ──► G1 encoder ──► bottleneck features
                                      │
                              GradReverse(α) ──► DomainDisc ──► BCE loss

Encoder is updated to FOOL the discriminator → domain-invariant features
```

`DomainDisc` is a 3-layer MLP (512·8·8 → 1024 → 512 → 1) with `AdaptiveAvgPool` and dropout (p=0.4).

---

## Project Structure

```
image-inpainting/
│
├── 01_data_processing_analysis.ipynb   # EDA, preprocessing, DataLoaders
├── 02_model_training.ipynb             # Full model — architecture + training
├── 03_checkpoint_reconstruction.ipynb  # Post-training evaluation & plots
├── 04_baseline_training.ipynb          # Single-stage U-Net for comparison
│
├── data/
│   ├── celeba/img_align_celeba/        # CelebA face images (.jpg)
│   ├── ffhq/images256x256/             # FFHQ high-res face images
│   ├── places/images/                  # Places365 scene images
│   ├── masks/irregular_masks/          # Freeform mask .png files
│   ├── cache/                          # nb1_artefacts.json + full_model_results.pkl
│   ├── processed/                      # 128px cached PNGs + _edge.npy files
│   └── splits/                         # train/val/test split JSONs
│
├── checkpoints/                        # Full model .pth checkpoints
├── checkpoints_baseline/               # Baseline model .pth checkpoints
│
└── results/                            # Generated plots and visualisations
    ├── training_curves.png
    ├── full_training_psnr_ssim.png
    ├── full_training_by_bin.png
    ├── final_summary_by_bin.png
    ├── best_checkpoint_qualitative.png
    ├── metrics_by_bin.png
    ├── ablation.png
    ├── qualitative_comparison.png
    └── comparison_full_vs_baseline.png
```

---


## Usage

### Step 1 — Data Processing (Notebook 1)

```bash
jupyter notebook 01_data_processing_analysis.ipynb
# Scans datasets, creates train/val/test splits (80/10/10),
# generates mask statistics, saves nb1_artefacts.json
```

**Output:**
- `data/cache/nb1_artefacts.json` — split paths, normalisation stats
- `data/processed/{train,val,test}/` — 128×128 cached PNGs + edge maps
- Split sizes: Train 162,079 | Val 20,259 | Test 20,261

### Step 2 — Full Model Training (Notebook 2)

```bash
jupyter notebook 02_model_training.ipynb
# Run all cells; training starts at the train() call
# ~75,000 iterations; checkpoints saved every 10k iters
```

Key configuration (in `CFG` dict):

```python
CFG = dict(
    lr_g=2e-4, lr_d=2e-5,          # Generator/discriminator LRs
    beta1=0.001, beta2=0.9,         # Adam betas (low β₁ for stability)
    n_iter=75_000,                  # Total training iterations
    mixed_prec=True,                # AMP enabled — ~1 it/s on RTX 3060
    lw_rec=10.0,                    # Reconstruction (L1) weight
    lw_perc=0.1,                    # Perceptual (VGG) weight
    lw_style=50.0,                  # Style (Gram) weight — NaN-safe value
    lw_adv=0.1,                     # Adversarial weight
    swin_window=4, swin_heads=4,    # Swin hyperparameters
    swin_depth=2, swin_mlp=4.0,
)
```

### Step 3 — Checkpoint Evaluation (Notebook 3)

```bash
jupyter notebook 03_checkpoint_reconstruction.ipynb
# Standalone — evaluates all saved checkpoints
# Generates 4 publication-quality plots in results/
```

### Step 4 — Baseline Comparison (Notebook 4)

```bash
jupyter notebook 04_baseline_training.ipynb
# Trains single-stage U-Net for ablation comparison
# Generates comparison_full_vs_baseline.png
```

---

## Training Details

### Loss Function

The generator optimises a weighted sum of five losses:

```
L_total = 10.0 · L_rec   (pixel-wise L1 on final output)
        +  0.1 · L_perc  (VGG-16 perceptual — float32 to prevent NaN)
        + 50.0 · L_style (Gram matrix — normalised by C·H·W)
        + 10.0 · L_feat  (discriminator feature matching, both D1 + D2)
        +  0.1 · L_adv   (hinge adversarial loss, both D1 + D2)
```

The VGG perceptual/style loss always runs in **float32** even when AMP is enabled — this prevents float16 NaN in Gram matrices.

The discriminator uses **hinge loss**:
```
L_D = E[relu(1 − D(real))] + E[relu(1 + D(fake))]
```

### Optimiser & Scheduler

```python
# Generator: Adam with warm-up + cosine annealing
opt_g  = Adam(gen_params,  lr=2e-4, betas=(0.001, 0.9), weight_decay=1e-4)
sched_g = LambdaLR(opt_g, lr_lambda)   # linear warm-up (2k iters) + cosine decay

# Discriminator: Adam with cosine annealing (no warm-up)
opt_d  = Adam(disc_params, lr=2e-5, betas=(0.001, 0.9))
sched_d = CosineAnnealingLR(opt_d, T_max=75_000, eta_min=1e-6)
```


---

## Mask Generation

Masks are generated procedurally using a combination of **random strokes** and **ellipses**:

```python
class MaskGenerator:
    # Stroke: random walk with angular perturbation, variable brush width
    # Ellipse: random centre, semi-axes, rotation
    # Rejection sampling until mask coverage ∈ [lo, hi]
    def generate(lo=0.10, hi=0.50) → np.ndarray  # uint8 {0,255}
```

Coverage bins used for evaluation: `10-20%`, `20-30%`, `30-40%`, `40-50%`.

---

## Evaluation Metrics

| Metric | Description | Notes |
|--------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better; computed on `[0,1]` denormalised images |
| **SSIM** | Structural Similarity Index | Higher is better; `data_range=1.0` |
| **FID** | Fréchet Inception Distance | Lower is better; requires `torch-fidelity` |

FID is computed every 10 validation steps (every 50k training iters) using 20 batches.

---

## Hardware & Training Time

| Component | Spec                           |
|-----------|--------------------------------|
| GPU | NVIDIA GeForce RTX 3060 Laptop |
| VRAM | 6.4 GB                         |
| PyTorch | 2.8.0+cu128                    |
| Full model train time | ~17–18 h (75k iters)           |
| Baseline train time | ~6–7 h (60k iters)             |


---

## Notebook Summary

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `01_data_processing_analysis.ipynb` | EDA, preprocessing, split creation | `nb1_artefacts.json`, processed cache |
| `02_model_training.ipynb` | Full model definition, training loop, evaluation | `checkpoints/ckpt_*.pth`, `results/` |
| `03_checkpoint_reconstruction.ipynb` | Post-hoc checkpoint evaluation & plotting | 4 publication plots, `full_model_results.pkl` |
| `04_baseline_training.ipynb` | Baseline U-Net training & comparison | `checkpoints_baseline/`, comparison plot |

---

## Technical References

- **Swin Transformer**: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
- **BIFPN**: Tan et al., "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020
- **Partial Convolutions / Irregular Masks**: Liu et al., "Image Inpainting for Irregular Holes Using Partial Convolutions", ECCV 2018
- **CTSDG**: Guo et al., "Image Inpainting via Conditional Texture and Structure Dual Generation", ICCV 2021
- **DANN (Gradient Reversal)**: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
- **PatchGAN**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017
- **VGG Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer", ECCV 2016

---


<div align="center">

*Built with PyTorch · Trained on RTX 3060 · CelebA + FFHQ + Places365*

</div>
