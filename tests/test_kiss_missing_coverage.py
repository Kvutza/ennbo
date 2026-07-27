from __future__ import annotations

import numpy as np

from bazel.audit_wheel import audit_wheel
from ennx.quantization import quantize_fp4_e2m1, quantize_int4
from ennx.turbo.config.enn_distance_metric import ENNDistanceMetric
from ennx.turbo.config.multi_tr_config import MultiTRConfig


def test_audit_wheel_is_callable():
    assert callable(audit_wheel)


def test_quantizers_pack_pairs_and_pad_odd_inputs():
    values = np.array([0.0, 1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(quantize_int4(values), np.array([0x10, 0x02]))
    np.testing.assert_array_equal(
        quantize_fp4_e2m1(values),
        np.array([0x20, 0x04]),
    )


def test_distance_metric_values_are_stable():
    assert ENNDistanceMetric.SQUARED_L2.value == "squared_l2"
    assert ENNDistanceMetric.COSINE.value == "cosine"


def test_multi_tr_length_properties_delegate_to_length_config():
    config = MultiTRConfig()

    assert config.length_init == config.length.length_init
    assert config.length_min == config.length.length_min
    assert config.length_max == config.length.length_max
