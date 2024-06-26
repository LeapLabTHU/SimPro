# SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed Semi-Supervised Learning

This repository contains the Pytorch implementation of the ICML 2024 paper "SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed Semi-Supervised Learning".

> **SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed Semi-Supervised Learning** <br> > [Chaoqun Du](https://scholar.google.com/citations?user=0PSKJuYAAAAJ&hl=en),
> [Yizeng Han](https://yizenghan.top/),
> [Gao Huang](https://www.gaohuang.net)

[![arXiv](https://img.shields.io/badge/arxiv-SimPro-blue)](https://arxiv.org/abs/2402.13505)

## Introduction

<p align="center">
    <img src="figures/fig1.png" width= "500" alt="fig1" />
</p>

## Method

<p align="center">
    <img src="figures/fig2.png" alt="fig1" />
</p>

## Get Started

### Requirements

### Training

By default, we use 1 RTX3090 GPU for CIFAR/STL10/ImageNet-32*32 datasets and 1 A100 GPU (40G) for ImageNet-64*64 dataset.

```[bash]
bash sh/${method}.sh ${dataset} ${exp_index}


```

## Citation

If you find this code useful, please consider citing our paper:

```[tex]
@inproceedings{du2024simpro,
    title={SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed Semi-Supervised Learning},
    author={Chaoqun Du and Yizeng Han and Gao Huang},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=NbOlmrB59Z}
}
```

## Contact

If you have any questions, please feel free to contact the authors. Chaoqun Du: <dcq20@mails.tsinghua.edu.cn>.

## Acknowledgement

Our code is based on the ACR (Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency Is All You Need).
