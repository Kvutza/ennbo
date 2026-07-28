from __future__ import annotations

from enum import Enum, auto


class ENNIndexDriver(Enum):
    FLAT = auto()
    USEARCH = auto()
    BPANN_DISK = auto()
    METAL = auto()
    OPENCL = auto()


# Canonical strings for Rust (model and optimizer both accept lowercase)
ENN_INDEX_DRIVER_TO_RUST: dict[ENNIndexDriver, str] = {
    ENNIndexDriver.FLAT: "exact",
    ENNIndexDriver.USEARCH: "usearch",
    ENNIndexDriver.BPANN_DISK: "bpann_disk",
    ENNIndexDriver.METAL: "metal",
    ENNIndexDriver.OPENCL: "opencl",
}
