import unittest

import torch

from vpf.adapters.mmdetection import MMDET_VPF
from vpf.adapters.mmsegmentation import MMSEG_VPF
from vpf.models import VPF


SMALL_MODEL = dict(depths=(1, 1, 1, 1), dims=(32, 64, 128, 256), img_size=32)


class VPFSmokeTest(unittest.TestCase):
    def test_classification_forward(self):
        model = VPF(num_classes=7, **SMALL_MODEL).eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 7))

    def test_detection_adapter_forward(self):
        model = MMDET_VPF(**SMALL_MODEL).eval()
        with torch.inference_mode():
            outputs = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [(1, 32, 8, 8), (1, 64, 4, 4), (1, 128, 2, 2), (1, 256, 1, 1)],
        )

    def test_segmentation_adapter_registration(self):
        self.assertEqual(MMSEG_VPF.__name__, "MMSEG_VPF")


if __name__ == "__main__":
    unittest.main()
