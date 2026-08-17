"""MMDetection registration for VPF."""

from mmdet.registry import MODELS

from ._common import OpenMMLabVPF


@MODELS.register_module()
class MMDET_VPF(OpenMMLabVPF):
    """VPF backbone registered in MMDetection."""


__all__ = ["MMDET_VPF"]
