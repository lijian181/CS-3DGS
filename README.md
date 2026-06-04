# CS-3DGS

**Compact Semantic 3D Gaussian Splatting via Adaptive Background Suppression and Budget-Controlled Densification**

CS-3DGS is a compact semantic 3D Gaussian Splatting framework designed to reduce redundant Gaussian primitives while preserving rendering quality and semantic consistency.

Semantic 3DGS methods usually inherit the densification strategy of vanilla 3DGS, where Gaussian cloning and splitting are mainly driven by RGB reconstruction errors. This may introduce redundant Gaussians in stable background regions and increase the cost of semantic feature maintenance. CS-3DGS addresses this problem by introducing foreground-background-aware redundancy control into semantic 3DGS training and post-processing.

## Overview

CS-3DGS first generates foreground-background priors from 2D semantic cues and accumulates them into Gaussian-level statistics. These statistics are used to guide adaptive background suppression, post-training safe deletion, and budget-controlled densification. The method aims to build a more compact semantic 3DGS representation while maintaining the spatial and semantic structure of Gaussians.

## Key Features

- Foreground-background prior generation for semantic 3DGS
- Gaussian-level foreground, background, uncertainty, and visibility statistics
- Adaptive suppression of redundant background Gaussians during training
- Post-training safe deletion of low-contribution stable-background Gaussians
- Budget-controlled clone and split densification
- Compatibility with attribute-level compression tools

## Pretrained Models & Weights

CS-3DGS relies on the following pretrained models for foreground-background prior generation and semantic understanding. Please download the corresponding weights and place them in the designated directories.

| Model | Weights File | Source / Download Link | Usage |
|-------|--------------|------------------------|--------|
| **BiRefNet** | `BiRefNet-general-epoch_244.pth` | [HuggingFace Hub](https://huggingface.co/ZhengPeng7/BiRefNet) or [Official GitHub](https://github.com/ZhengPeng7/BiRefNet) | High-resolution dichotomous image segmentation for foreground probability map |
| **SAM (Segment Anything Model)** | `sam_vit_h_4b8939.pth` | [SAM Official Repository](https://github.com/facebookresearch/segment-anything) | Object boundary constraints and mask proposals for region refinement |
| **SAM 2** | `sam2_hiera_large.pt` | [SAM 2 Official Repository](https://github.com/facebookresearch/segment-anything-2) | Enhanced mask generation for more robust foreground/background separation |





## Code

The code will be released after the paper is published.
