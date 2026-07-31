from __future__ import annotations

import numpy as np

from ennx.quantization import quantize_fp4_e2m1, quantize_int4


def test_quantizers_pack_pairs_and_pad_odd_inputs():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(quantize_int4(x), np.array([0x10, 0x02]))
    np.testing.assert_array_equal(quantize_fp4_e2m1(x), np.array([0x20, 0x04]))
