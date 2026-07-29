from __future__ import annotations

from ._lazy import lazy_getattr

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "EpistemicNearestNeighbors": (".ennx.enn_class", "EpistemicNearestNeighbors"),
    "ENNStatefulFitter": (".ennx.enn_fitter", "ENNStatefulFitter"),
    "experimental": (".experimental", "experimental"),
    "create_optimizer": (".turbo.rust_optimizer", "create_optimizer"),
    "create_optimizer_enn": ("._rust", "create_optimizer_enn"),
    "create_optimizer_zero": ("._rust", "create_optimizer_zero"),
    "create_optimizer_lhd": ("._rust", "create_optimizer_lhd"),
    "Telemetry": (".turbo.types.telemetry", "Telemetry"),
    "OptimizerConfig": (".turbo.optimizer_config", "OptimizerConfig"),
    "turbo_one_config": (".turbo.optimizer_config", "turbo_one_config"),
    "turbo_zero_config": (".turbo.optimizer_config", "turbo_zero_config"),
    "turbo_enn_config": (".turbo.optimizer_config", "turbo_enn_config"),
    "lhd_only_config": (".turbo.optimizer_config", "lhd_only_config"),
    "TurboTRConfig": (".turbo.config.trust_region", "TurboTRConfig"),
    "MorboTRConfig": (".turbo.config.trust_region", "MorboTRConfig"),
    "NoTRConfig": (".turbo.config.trust_region", "NoTRConfig"),
    "CandidateRV": (".turbo.optimizer_config", "CandidateRV"),
    "InitStrategy": (".turbo.optimizer_config", "InitStrategy"),
    "AcqType": (".turbo.optimizer_config", "AcqType"),
    "quantize_int4": (".quantization", "quantize_int4"),
    "quantize_fp4_e2m1": (".quantization", "quantize_fp4_e2m1"),
}



def __getattr__(name: str):
    return lazy_getattr(
        name=name,
        module_name=__name__,
        package=__package__,
        mapping=_LAZY_ATTRS,
        extra="`pip install 'ennx[with-deps]'`",
    )


__all__: list[str] = [
    "AcqType",
    "CandidateRV",
    "ENNStatefulFitter",
    "EpistemicNearestNeighbors",
    "InitStrategy",
    "MorboTRConfig",
    "NoTRConfig",
    "OptimizerConfig",
    "Telemetry",
    "TurboTRConfig",
    "create_optimizer",
    "create_optimizer_enn",
    "create_optimizer_lhd",
    "create_optimizer_zero",
    "experimental",
    "lhd_only_config",
    "quantize_fp4_e2m1",
    "quantize_int4",
    "turbo_enn_config",
    "turbo_one_config",
    "turbo_zero_config",
]
