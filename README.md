# ptmq-pytorch -> ptmq_w's implementation

PyTorch implementation of "PTMQ: Post-training Multi-Bit Quantization of Neural Networks (Xu et al., AAAI 2024)"

[[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29553)

---

## Getting Started

Running PTMQ

```bash
python run_ptmq.py --config configs/[config_file].yaml
```

Create your own configuration file in the `configs` directory.

---

## Useful Commands

### Initial Setup for Cloud GPUs ([runpod.io](https://runpod.io?ref=9t3u4v13))

```bash
# create virtual environment and install dependencies
python -m venv iris
source iris/bin/activate
pip install --upgrade pip
pip install torch torchvision easydict PyYAML scipy gdown

# download resnet18 weights
cd ~/dev/ptmq-pytorch
python
import torch
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
torch.save(resnet18.state_dict(), "resnet18_weights.pth")
```

```bash
# login for wandb
pip install wandb
wandb login
# enter wandb API key from https://wandb.ai/authorize
```

### Downloading Datasets

Use `imagenet-mini/train` for calibration, and `imagenet/val` (the full ImageNet1K validation set) for evaluation.

#### Mini-ImageNet from Kaggle

```bash
pip install kaggle
cd ~/dev
mkdir -p ~/dev/kaggle # add kaggle.json with {"username":"xxx","key":"xxx"} here
chmod 600 ~/dev/kaggle/kaggle.json
kaggle datasets download -d ifigotin/imagenetmini-1000
apt-get update && apt-get install unzip
unzip ~/dev/imagenetmini-1000.zip -d ~/dev
```

#### ImageNet Validation Dataset

```bash
# download imagenet validation dataset (from public google drive)
mkdir -p ~/dev/imagenet/val
cd imagenet/val
gdown https://drive.google.com/uc?id=11omFedOvjslBRMFc-lrM3n2t0xP99FXB -O ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### `wandb` setup

```bash
pip install wandb
wandb login # enter API key
```

---

## Overview

PTMQ (post-training multi-bit quantization) is a post-training quantization method that performs **block-level activation quantization** with multiple bit-widths.

In a sense, it can be viewed as a knowledge distillation method, where the higher-precision features are used to guide the quantization of lower-precision features.

For starters, we can think of the target inference model to be W3A3. PTMQ has separate strategies in order to ensure that the weights and activations are better quantized. These are all employed during the block reconstruction phase, and weights and activations are optimized with a reconstruction loss (GD loss) and round loss, respectively.

**Weights** are better quantized using rounding optimization via AdaRound (Nagel et al., ICML 2020). This is done by minimizing the quantization error of the weights.

**Activations** are better quantized by using a multi-bit feature mixer, which is 3 separate quantizers for low, medium, and high bit-widths. We learn activation step sizes (Esser et al., ICLR 2020) to minimize the activation quantization error, via a group-wise distillation loss.

The novelty of this model is that through block reconstruction, we can quickly and efficiently quantize a full-precision model to multiple bit-widths, which can be flexibly be deployed based on the given hardware constraints in real-time.

---

## Reproducing Results

### ResNet-18

- iterations: 5000
- block reconstruction hyperparameters:
  - batch_size: 32
  - scale_lr: 4.0e-5
  - warm_up: 0.2
  - weight: 0.01
  - iters: 5000 #20000
  - b_range: [20, 2]
  - keep_gpu: True
  - round_mode: learned_hard_sigmoid
  - mixed_p: 0.5
- ptmq hyperparameters:
  - lambda1: 0.4
  - lambda2: 0.3
  - lambda3: 0.3
  - mixed_p: 0.5
  - gamma1: 100
  - gamma2: 100
  - gamma3: 100
- first and last layer weights: 8-bit
- last layer activations: 8-bit
- low, medium, high bit-widths for each precision (not mentioned in paper)
  - W3A3: (l, m, h) = (3, 4, 5)
  - W4A4: (l, m, h) = (4, 5, 6)
  - W5A5: (l, m, h) = (5, 6, 7)
  - W6A6: (l, m, h) = (6, 7, 8)

`ResNet-18`
| Info | Precision Type | GPU | Time (min) | W3A3 | W4A4 | W5A5 | W6A6 | **W32A32** |
| ----- | ------------- | --- | ---------- | ---- | ---- | ---- | ---- | ------ |
| Paper | Mixed | Nvidia 3090 | 100 | 64.02 | 67.57 | 69.00 | 70.23 | **71.08** |
| Our Code | Uniform | Nvidia A40 | 14.41 | 63.47 | 67.59 | 69.02 | 69.50 | **71.08** |

We compare how relative bit-precision affects our desired performance precision.

| Precision Type | Precision | Mixed-bit (l,m,h) | Top-1 (%) |
| -------------- | --------- | ----------------- | --------- |
| - | W32A32 | - | 71.08 |
| Mixed | W5A5 | **Baseline** (unknown) | 69.00 |
| Uniform | W5A5 | (3, 4, **5**) | 65.95 |
| Uniform | W5A5 | (4, **5**, 6) | 67.90 |
| Uniform | W5A5 | (**5**, 6, 7) | 69.02 |



### Vision Transformer

- Paper results
  - ViT-S/224/16 W8A8 - top1 = 78.16%

---

## Implementation

- we build off of the QDrop source code
- key differences from original source code
  - `quant_module.py`: add multi-bit feature for forward (`QuantLinear`, `QuantConv2D`, `QuantBlock`)
  - `ptmq_recon.py` reconstruction for ptmq, with MFM and GD loss (MSE for layer reconstruction, perhaps remove?)

---

## TODO

- [x] PTMQ - Key Contributions
  - [x] Multi-bit Feature Mixer (MFM)
  - [x] Group-wise Distill Loss (GD-Loss)
- [x] Fundamental Tools
  - [x] Rounding-based quantization (AdaRound)
  - [x] BatchNorm folding
- [x] Quantization Modules
  - [x] Layer quantization
  - [x] Block quantization
  - [x] Model quantization
- [x] Reconstruction
  - [x] Block Reconstruction
- [ ] 🔥PTMQ - Sanity Test
  - [x] CNN - ResNet-18
  - [ ] 🔥Transformer - ViT
- [ ] Preliminary Results
  - [ ] PTMQ verification
    - [x] 🔥CNN - ResNet-18
    - [ ] 🔥Transformer - ViT
- [ ] PTMQ - Mixed-Precision Quantization
  - [ ] Pareto Frontier search (ImageNet and ResNet-18)
- [ ] Final Overview
  - [ ] verify on most/all experiments
  - [ ] (partially) reproduce results
