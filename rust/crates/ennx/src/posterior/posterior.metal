#include <metal_stdlib>
using namespace metal;

struct Params {
    uint queries;
    uint neighbors;
    uint metrics;
    float epistemic_scale;
    float aleatoric_scale;
};

template <uint Lanes>
inline void posterior(
    device const float* distances,
    device const uint* indices,
    device const float* outcomes,
    device const float* y_scale,
    device float* mu,
    device float* se,
    constant Params& params,
    uint gid
) {
    uint count = params.queries * params.metrics;
    if (gid >= count) return;
    uint query = gid / params.metrics;
    uint metric = gid - query * params.metrics;
    float weights[Lanes];
    float means[Lanes];
    for (uint lane = 0; lane < Lanes; ++lane) {
        weights[lane] = 0.0f;
        means[lane] = 0.0f;
    }
    uint base = query * params.neighbors;
    for (uint j = 0; j < params.neighbors; j += Lanes) {
        for (uint lane = 0; lane < Lanes; ++lane) {
            uint at = j + lane;
            if (at < params.neighbors) {
                float variance = 1.0e-9f
                    + params.epistemic_scale * distances[base + at]
                    + params.aleatoric_scale;
                float weight = 1.0f / variance;
                weights[lane] += weight;
                means[lane] = fma(
                    weight,
                    outcomes[indices[base + at] * params.metrics + metric],
                    means[lane]
                );
            }
        }
    }
    float weight_sum = weights[0];
    float weighted_mean = means[0];
    for (uint lane = 1; lane < Lanes; ++lane) {
        weight_sum += weights[lane];
        weighted_mean += means[lane];
    }
    float inverse = 1.0f / weight_sum;
    mu[gid] = weighted_mean * inverse;
    se[gid] = sqrt(max(inverse, 1.0e-9f)) * y_scale[metric];
}

kernel void posterior_1(
    device const float* distances [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* outcomes [[buffer(2)]],
    device const float* y_scale [[buffer(3)]],
    device float* mu [[buffer(4)]],
    device float* se [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    posterior<1>(distances, indices, outcomes, y_scale, mu, se, params, gid);
}

kernel void posterior_2(
    device const float* distances [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* outcomes [[buffer(2)]],
    device const float* y_scale [[buffer(3)]],
    device float* mu [[buffer(4)]],
    device float* se [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    posterior<2>(distances, indices, outcomes, y_scale, mu, se, params, gid);
}

kernel void posterior_4(
    device const float* distances [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* outcomes [[buffer(2)]],
    device const float* y_scale [[buffer(3)]],
    device float* mu [[buffer(4)]],
    device float* se [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    posterior<4>(distances, indices, outcomes, y_scale, mu, se, params, gid);
}
