"""MMSegmentation registration for VPF."""

from mmseg.registry import MODELS

from ._common import OpenMMLabVPF


@MODELS.register_module()
class MMSEG_VPF(OpenMMLabVPF):
    """VPF backbone registered in MMSegmentation."""


__all__ = ["MMSEG_VPF"]
