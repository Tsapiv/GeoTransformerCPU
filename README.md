# Geometric Transformer for Fast and Robust Point Cloud Registration

PyTorch implementation of the paper:

[Geometric Transformer for Fast and Robust Point Cloud Registration](https://arxiv.org/abs/2202.06688).

[Zheng Qin](https://scholar.google.com/citations?user=DnHBAN0AAAAJ), [Hao Yu](https://scholar.google.com/citations?user=g7JfRn4AAAAJ), Changjian Wang, [Yulan Guo](https://scholar.google.com/citations?user=WQRNvdsAAAAJ), Yuxing Peng, and [Kai Xu](https://scholar.google.com/citations?user=GuVkg-8AAAAJ).

## Introduction

![](assets/teaser.png)

## Installation

Please use the following command for installation.

```bash
poetry install
```

## Installation as package
Use pip to install:
```bash
pip install git+https://github.com/Tsapiv/GeoTransformerCPU.git
```
or poetry (add following to your project's `pyproject.toml` ):
```toml
[tool.poetry.dependencies]
# ...
geotransformer = { git = "https://github.com/Tsapiv/GeoTransformerCPU.git" }
# ...
[[tool.poetry.source]]
name     = "pytorch-cpu"
priority = "explicit"
url      = "https://download.pytorch.org/whl/cpu"
```

## Citation

```bibtex
@inproceedings{qin2022geometric,
    title={Geometric Transformer for Fast and Robust Point Cloud Registration},
    author={Zheng Qin and Hao Yu and Changjian Wang and Yulan Guo and Yuxing Peng and Kai Xu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={11143-11152}
}
```

## Acknowledgements

- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [RPMNet](https://github.com/yewzijian/RPMNet)
- [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences)
- [huggingface-transformer](https://github.com/huggingface/transformers)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
