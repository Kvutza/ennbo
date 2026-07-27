from __future__ import annotations

PRODUCTION_SURROGATE_CONFIG = "GPSurrogateConfig"

PRODUCTION_MODULES = (
    "optimizer",
    "optimizer_generate",
    "morbo_trust_region",
    "turbo_trust_region",
    "no_trust_region",
    "turbo_gp",
    "turbo_gp_fit",
    "turbo_gp_base",
    "turbo_gp_noisy",
    "turbo_utils",
    "turbo_utils_core",
    "turbo_utils_gp",
    "turbo_utils_incumbent",
    "turbo_utils_perturb",
    "turbo_utils_tr",
    "turbo_optimizer_utils",
    "sampling",
    "components.gp_surrogate",
    "components.builder",
    "components.acquisition",
    "components.thompson_acq_optimizer",
    "components.ucb_acq_optimizer",
    "components.pareto_acq_optimizer",
    "components.chebyshev_incumbent_selector",
    "components.incumbent_selector",
    "components.incumbent_tracker",
    "strategies.turbo_hybrid_strategy",
)

TEST_ONLY_MODULES = (
    "components.no_surrogate",
    "components.random_acq_optimizer",
    "strategies.lhd_only_strategy",
    "components.surrogates",
    "components.surrogate_result",
    "components.posterior_result",
    "components.incumbent_selector_protocol",
)
