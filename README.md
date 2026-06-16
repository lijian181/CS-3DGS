# CS-3DGS

**Compact Semantic 3D Gaussian Splatting via Adaptive Background Suppression and Budget-Controlled Densification**

CS-3DGS 是一种紧凑语义 3D Gaussian Splatting 框架，旨在减少冗余 Gaussian 基元，同时保持渲染质量和语义一致性。

语义 3DGS 方法通常继承 vanilla 3DGS 的稠密化策略，其中 Gaussian 的克隆和分裂主要由 RGB 重建误差驱动。这可能会在稳定背景区域引入冗余 Gaussians，并增加语义特征维护的成本。CS-3DGS 通过在语义 3DGS 训练和后处理中引入前景-背景感知的冗余控制来解决这一问题。

## 概述

CS-3DGS 首先从二维语义线索中生成前景-背景先验，并将其累积为 Gaussian 级别的统计信息。这些统计信息用于指导自适应背景抑制、训练后安全删除和预算控制稠密化。该方法旨在构建更加紧凑的语义 3DGS 表示，同时保持 Gaussians 的空间结构和语义结构。

## 主要特性

- 面向语义 3DGS 的前景-背景先验生成
- Gaussian 级别的前景、背景、不确定性和可见性统计
- 训练过程中对冗余背景 Gaussians 的自适应抑制
- 训练后对低贡献稳定背景 Gaussians 的安全删除
- 预算控制的克隆和分裂稠密化
- 兼容属性级压缩工具

## 预训练模型与权重

CS-3DGS 依赖以下预训练模型进行前景-背景先验生成和语义理解。请下载相应权重，并将其放置在指定目录中。

| 模型 | 权重文件 | 来源 / 下载链接 | 用途 |
|-------|--------------|------------------------|--------|
| **BiRefNet** | `BiRefNet-general-epoch_244.pth` | [HuggingFace Hub](https://huggingface.co/ZhengPeng7/BiRefNet) 或 [Official GitHub](https://github.com/ZhengPeng7/BiRefNet) | 用于生成前景概率图的高分辨率二分图像分割 |
| **SAM (Segment Anything Model)** | `sam_vit_h_4b8939.pth` | [SAM Official Repository](https://github.com/facebookresearch/segment-anything) | 用于区域细化的目标边界约束和掩码提议 |
| **SAM 2** | `sam2_hiera_large.pt` | [SAM 2 Official Repository](https://github.com/facebookresearch/segment-anything-2) | 用于更稳健前景/背景分离的增强掩码生成 |

## 环境配置

本项目的运行环境与原版 3D Gaussian Splatting（3DGS）环境保持一致。请先按照原版 3DGS 的环境配置方式安装 CUDA、PyTorch 以及相关依赖。

如果已经能够正常运行原版 3DGS，本项目通常可以直接在相同环境下运行。

## 数据与预训练模型下载

为了方便复现实验，本项目所需的数据和预训练模型权重已打包为 `data.7z`。

国内用户可以通过夸克网盘下载：

```text
文件名：data.7z
链接：https://pan.quark.cn/s/52b187e351dd
提取码：gyg8
