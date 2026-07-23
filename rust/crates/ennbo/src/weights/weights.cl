#define THREADS 256u
#define MAX_NEIGHBORS 2048u

typedef struct {
    uint offset;
    uint length;
    uint bits;
    float quantization_scale;
    float metric_scale;
    float weight;
} Block;

typedef struct {
    uint row_bytes;
    uint observations;
    uint candidates;
    uint blocks;
    uint neighbors;
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
    uint acquisition;
} Params;

inline float qvalue(__global const uchar* row, uint byte_base, uint element, uint bits) {
    if (bits == 4u) {
        uchar byte = row[byte_base + element / 2u];
        uint shift = (element & 1u) * 4u;
        return (float)((byte >> shift) & 0x0fu);
    }
    return (float)(row[byte_base + element]);
}

__kernel void score_weight_neighbors(
    __global const uchar* observations,
    __global const float* outcomes,
    __global const uchar* candidates,
    __global const Block* blocks,
    __global float* scores,
    __global const float* thompson_draws,
    Params params
) {
    uint candidate_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (candidate_index >= params.candidates) {
        return;
    }

    __local float partials[THREADS];
    __local float nearest_distances[MAX_NEIGHBORS];
    __local uint nearest_indices[MAX_NEIGHBORS];

    if (thread_index == 0u) {
        for (uint k = 0u; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global const uchar* candidate = candidates + ((ulong)candidate_index) * ((ulong)params.row_bytes);
    for (uint observation_index = 0u; observation_index < params.observations; ++observation_index) {
        __global const uchar* observation = observations + ((ulong)observation_index) * ((ulong)params.row_bytes);
        float accum = 0.0f;
        uint byte_base = 0u;
        for (uint block_index = 0u; block_index < params.blocks; ++block_index) {
            Block block = blocks[block_index];
            float scale = block.quantization_scale;
            for (uint element = thread_index; element < block.length; element += THREADS) {
                float a = qvalue(candidate, byte_base, element, block.bits) * scale;
                float b = qvalue(observation, byte_base, element, block.bits) * scale;
                float delta = a - b;
                accum = fma(delta, delta * block.weight, accum);
            }
            byte_base += (block.bits == 4u) ? ((block.length + 1u) / 2u) : block.length;
        }
        partials[thread_index] = accum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (thread_index == 0u) {
            float distance = partials[0];
            uint insert_at = params.neighbors;
            for (uint k = 0u; k < params.neighbors; ++k) {
                if (distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && observation_index < nearest_indices[k])) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at < params.neighbors) {
                for (uint k = params.neighbors - 1u; k > insert_at; --k) {
                    nearest_distances[k] = nearest_distances[k - 1u];
                    nearest_indices[k] = nearest_indices[k - 1u];
                }
                nearest_distances[insert_at] = distance;
                nearest_indices[insert_at] = observation_index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_index == 0u) {
        float weight_sum = 0.0f;
        float weighted_outcome = 0.0f;
        for (uint k = 0u; k < params.neighbors; ++k) {
            float variance = 1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / fmax(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_outcome += weight * outcomes[nearest_indices[k]];
        }
        float mean = weighted_outcome / fmax(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / fmax(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * thompson_draws[candidate_index];
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}
