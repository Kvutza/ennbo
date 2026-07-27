from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .tr_length_config import TRLengthConfig


@dataclass(frozen=True)
class MultiTRConfig:
    num_regions: int = 4
    length: TRLengthConfig = TRLengthConfig()
    succ_tolerance: int = 3
    fail_tolerance: int = 5
    sharing_policy: Literal["shared", "nearest_center", "independent"] = "shared"

    @property
    def length_init(self) -> float:
        return self.length.length_init

    @property
    def length_min(self) -> float:
        return self.length.length_min

    @property
    def length_max(self) -> float:
        return self.length.length_max
