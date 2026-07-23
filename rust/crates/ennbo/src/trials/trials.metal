#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kMaxHistory = 16;

struct Leaf {
    uint byte_offset;
    uint element_offset;
    uint length;
    uint bits;
    float scale;
    float weight;
    uint whole;
    uint threshold;
};

struct Tile {
    uint leaf;
    uint start;
    uint length;
    uint pad;
};

struct Seed {
    uint low;
    uint high;
};

struct Params {
    uint row_bytes;
    uint history;
    uint candidates;
    uint leaves;
    uint tiles;
    uint neighbors;
    uint base_slot;
    uint trial_slot;
    uint acquisition;
    float epistemic_scale;
    float aleatoric_scale;
    float y_scale;
    float beta;
};

inline uint hash(Seed seed, uint element) {
    uint value = seed.low ^ element * 0x9e3779b9;
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= seed.high;
    value *= 0x846ca68b;
    return value ^ (value >> 15);
}

inline uint code_at(device const uchar* row, Leaf leaf, uint element) {
    if (leaf.bits == 4) {
        uchar byte = row[leaf.byte_offset + element / 2];
        return (byte >> ((element & 1u) * 4u)) & 0x0fu;
    }
    return row[leaf.byte_offset + element];
}

inline uint perturb(uint code, Seed seed, uint element, Leaf leaf) {
    uint random = hash(seed, element);
    uint amount = leaf.whole + uint((random >> 1u) < (leaf.threshold >> 1u));
    if (amount == 0) {
        return code;
    }
    uint max_code = (1u << leaf.bits) - 1u;
    if ((random & 1u) == 0u) {
        return code >= amount ? code - amount : min(code + amount, max_code);
    }
    return code + amount <= max_code ? code + amount : code >= amount ? code - amount : 0u;
}

kernel void distance_trials(
    device const uchar* rows [[buffer(0)]],
    device const uint* history_slots [[buffer(1)]],
    device const Seed* seeds [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const Tile* tiles [[buffer(4)]],
    device float* partials_out [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint candidate_group = group_index.x / params.tiles;
    uint tile_index = group_index.x - candidate_group * params.tiles;
    uint first_candidate = candidate_group * 2u;
    if (first_candidate >= params.candidates) {
        return;
    }
    bool has_second = first_candidate + 1u < params.candidates;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed first_seed = seeds[first_candidate];
    Seed second_seed = has_second ? seeds[first_candidate + 1u] : first_seed;
    device const uchar* base =
        rows + ulong(params.base_slot) * ulong(params.row_bytes);
    float first_distances[kMaxHistory];
    float second_distances[kMaxHistory];
    for (uint h = 0; h < params.history; ++h) {
        first_distances[h] = 0.0f;
        second_distances[h] = 0.0f;
    }

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += kThreads) {
            uint first = tile.start + local_byte * 2u;
            uchar base_byte = base[leaf.byte_offset + first_byte + local_byte];
            uint first_low = perturb(
                uint(base_byte & 0x0fu),
                first_seed,
                leaf.element_offset + first,
                leaf
            );
            uint first_high = 0u;
            if (first + 1u < leaf.length) {
                first_high = perturb(
                    uint(base_byte >> 4u),
                    first_seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            uint second_low = 0u;
            uint second_high = 0u;
            if (has_second) {
                second_low = perturb(
                    uint(base_byte & 0x0fu),
                    second_seed,
                    leaf.element_offset + first,
                    leaf
                );
                if (first + 1u < leaf.length) {
                    second_high = perturb(
                        uint(base_byte >> 4u),
                        second_seed,
                        leaf.element_offset + first + 1u,
                        leaf
                    );
                }
            }
            for (uint h = 0; h < params.history; ++h) {
                device const uchar* observation =
                    rows + ulong(history_slots[h]) * ulong(params.row_bytes);
                uchar observed = observation[leaf.byte_offset + first_byte + local_byte];
                float first_low_delta =
                    (float(first_low) - float(observed & 0x0fu)) * leaf.scale;
                first_distances[h] = fma(
                    first_low_delta,
                    first_low_delta * leaf.weight,
                    first_distances[h]
                );
                if (first + 1u < leaf.length) {
                    float first_high_delta =
                        (float(first_high) - float(observed >> 4u)) * leaf.scale;
                    first_distances[h] = fma(
                        first_high_delta,
                        first_high_delta * leaf.weight,
                        first_distances[h]
                    );
                }
                if (has_second) {
                    float second_low_delta =
                        (float(second_low) - float(observed & 0x0fu)) * leaf.scale;
                    second_distances[h] = fma(
                        second_low_delta,
                        second_low_delta * leaf.weight,
                        second_distances[h]
                    );
                    if (first + 1u < leaf.length) {
                        float second_high_delta =
                            (float(second_high) - float(observed >> 4u)) * leaf.scale;
                        second_distances[h] = fma(
                            second_high_delta,
                            second_high_delta * leaf.weight,
                            second_distances[h]
                        );
                    }
                }
            }
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += kThreads) {
            uint first_value = perturb(
                uint(base[leaf.byte_offset + element]),
                first_seed,
                leaf.element_offset + element,
                leaf
            );
            uint second_value = has_second
                ? perturb(
                    uint(base[leaf.byte_offset + element]),
                    second_seed,
                    leaf.element_offset + element,
                    leaf
                )
                : 0u;
            for (uint h = 0; h < params.history; ++h) {
                device const uchar* observation =
                    rows + ulong(history_slots[h]) * ulong(params.row_bytes);
                float first_delta =
                    (float(first_value) - float(observation[leaf.byte_offset + element]))
                    * leaf.scale;
                first_distances[h] = fma(
                    first_delta,
                    first_delta * leaf.weight,
                    first_distances[h]
                );
                if (has_second) {
                    float second_delta =
                        (float(second_value) - float(observation[leaf.byte_offset + element]))
                        * leaf.scale;
                    second_distances[h] = fma(
                        second_delta,
                        second_delta * leaf.weight,
                        second_distances[h]
                    );
                }
            }
        }
    }

    threadgroup float partials[kThreads];
    for (uint h = 0; h < params.history; ++h) {
        partials[thread_index] = first_distances[h];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = kThreads >> 1; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index == 0) {
            ulong offset =
                (ulong(first_candidate) * ulong(params.history) + ulong(h))
                * ulong(params.tiles)
                + ulong(tile_index);
            partials_out[offset] = partials[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (has_second) {
            partials[thread_index] = second_distances[h];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = kThreads >> 1; stride > 0; stride >>= 1) {
                if (thread_index < stride) {
                    partials[thread_index] += partials[thread_index + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (thread_index == 0) {
                ulong offset =
                    (ulong(first_candidate + 1u) * ulong(params.history) + ulong(h))
                    * ulong(params.tiles)
                    + ulong(tile_index);
                partials_out[offset] = partials[0];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void score_trials(
    device const float* partials_in [[buffer(0)]],
    device const float* outcomes [[buffer(1)]],
    device const float* draws [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint candidate_index = group_index.x;
    if (candidate_index >= params.candidates) {
        return;
    }
    threadgroup float partials[kThreads];
    threadgroup float nearest_distances[kMaxHistory];
    threadgroup uint nearest_indices[kMaxHistory];
    if (thread_index == 0) {
        for (uint k = 0; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < params.history; ++h) {
        float local = 0.0f;
        ulong base =
            (ulong(candidate_index) * ulong(params.history) + ulong(h))
            * ulong(params.tiles);
        for (uint tile = thread_index; tile < params.tiles; tile += kThreads) {
            local += partials_in[base + ulong(tile)];
        }
        partials[thread_index] = local;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = kThreads >> 1; stride > 0; stride >>= 1) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index == 0) {
            float distance = partials[0];
            uint insert_at = params.neighbors;
            for (uint k = 0; k < params.neighbors; ++k) {
                if (
                    distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && h < nearest_indices[k])
                ) {
                    insert_at = k;
                    break;
                }
            }
            if (insert_at < params.neighbors) {
                for (uint k = params.neighbors - 1; k > insert_at; --k) {
                    nearest_distances[k] = nearest_distances[k - 1];
                    nearest_indices[k] = nearest_indices[k - 1];
                }
                nearest_distances[insert_at] = distance;
                nearest_indices[insert_at] = h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        float weight_sum = 0.0f;
        float weighted_value = 0.0f;
        for (uint k = 0; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / max(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
        }
        float mean = weighted_value / max(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / max(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * draws[candidate_index];
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

kernel void pick_trial(
    device const float* scores [[buffer(0)]],
    device uint* choice [[buffer(1)]],
    constant Params& params [[buffer(2)]]
) {
    uint best = 0;
    float best_score = scores[0];
    for (uint index = 1; index < params.candidates; ++index) {
        float score = scores[index];
        if (score > best_score) {
            best = index;
            best_score = score;
        }
    }
    choice[0] = best;
}

kernel void write_trial(
    device uchar* rows [[buffer(0)]],
    device const Seed* seeds [[buffer(1)]],
    device const uint* choice [[buffer(2)]],
    device const Leaf* leaves [[buffer(3)]],
    device const Tile* tiles [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 group_index [[threadgroup_position_in_grid]]
) {
    uint tile_index = group_index.x;
    if (tile_index >= params.tiles) {
        return;
    }
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed seed = seeds[choice[0]];
    device const uchar* base =
        rows + ulong(params.base_slot) * ulong(params.row_bytes);
    device uchar* trial =
        rows + ulong(params.trial_slot) * ulong(params.row_bytes);

    if (leaf.bits == 4) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += kThreads) {
            uint first = tile.start + local_byte * 2u;
            uint low = perturb(
                code_at(base, leaf, first),
                seed,
                leaf.element_offset + first,
                leaf
            );
            uint high = 0;
            if (first + 1u < leaf.length) {
                high = perturb(
                    code_at(base, leaf, first + 1u),
                    seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            trial[leaf.byte_offset + first_byte + local_byte] = uchar(low | (high << 4u));
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += kThreads) {
            trial[leaf.byte_offset + element] = uchar(perturb(
                code_at(base, leaf, element),
                seed,
                leaf.element_offset + element,
                leaf
            ));
        }
    }
}
