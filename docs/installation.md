# Installation

The code was validated in the `vpf_det` environment with Python 3.9.23,
PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, timm 1.0.22, MMEngine 0.10.1,
MMCV 2.1.0, MMDetection 3.3.0, and MMSegmentation 1.2.2.

For the existing environment:

```bash
conda activate vpf_det
python -m pip install -e . --no-deps
```

Editable installation is required so task entry points and standalone analysis
tools can import the shared package in `src/vpf`.

For a fresh environment, install the correct CUDA build of PyTorch first, then
install one of the files under `requirements/`. Build MMCV using the official
OpenMMLab instructions if a compatible wheel is not available.
