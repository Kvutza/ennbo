from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ENNParams:
    k: int
    var_scale: float
