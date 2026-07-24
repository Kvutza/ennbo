#define THREADS 256u
#define MAX_HISTORY 16u

typedef struct {
    uint byte_offset;
    uint element_offset;
    uint length;
    uint bits;
    float scale;
    float weight;
    uint whole;
    uint threshold;
} Leaf;

typedef struct {
    uint leaf;
    uint start;
    uint length;
    uint pad;
} Tile;

typedef struct {
    uint low;
    uint high;
} Seed;

typedef struct {
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
} Params;

inline uint trial_hash(Seed seed, uint element) {
    uint value = seed.low ^ element * 0x9e3779b9u;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= seed.high;
    value *= 0x846ca68bu;
    return value ^ (value >> 15u);
}

inline uint code_at(__global const uchar* row, Leaf leaf, uint element) {
    if (leaf.bits == 4u) {
        uchar byte = row[leaf.byte_offset + element / 2u];
        return (byte >> ((element & 1u) * 4u)) & 0x0fu;
    }
    return row[leaf.byte_offset + element];
}

inline uint perturb_code(uint code, Seed seed, uint element, Leaf leaf) {
    uint random = trial_hash(seed, element);
    uint amount =
        leaf.whole + (uint)((random >> 1u) < (leaf.threshold >> 1u));
    if (amount == 0u) {
        return code;
    }
    uint max_code = (1u << leaf.bits) - 1u;
    if ((random & 1u) == 0u) {
        return code >= amount ? code - amount : min(code + amount, max_code);
    }
    return code + amount <= max_code
        ? code + amount
        : code >= amount ? code - amount : 0u;
}

__kernel void distance_trials(
    __global const uchar* rows,
    __global const uint* history_slots,
    __global const Seed* seeds,
    __global const Leaf* leaves,
    __global const Tile* tiles,
    __global float* partials_out,
    Params params
) {
    uint thread_index = get_local_id(0);
    uint group_index = get_group_id(0);
    uint candidate_group = group_index / params.tiles;
    uint tile_index = group_index - candidate_group * params.tiles;
    uint first_candidate = candidate_group * 2u;
    if (first_candidate >= params.candidates) {
        return;
    }
    int has_second = first_candidate + 1u < params.candidates;
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed first_seed = seeds[first_candidate];
    Seed second_seed = has_second ? seeds[first_candidate + 1u] : first_seed;
    __global const uchar* base =
        rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
    float first_distances[MAX_HISTORY];
    float second_distances[MAX_HISTORY];
    for (uint h = 0u; h < params.history; ++h) {
        first_distances[h] = 0.0f;
        second_distances[h] = 0.0f;
    }

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += THREADS) {
            uint first = tile.start + local_byte * 2u;
            uchar base_byte = base[leaf.byte_offset + first_byte + local_byte];
            uint first_low = perturb_code(
                (uint)(base_byte & 0x0fu),
                first_seed,
                leaf.element_offset + first,
                leaf
            );
            uint first_high = 0u;
            if (first + 1u < leaf.length) {
                first_high = perturb_code(
                    (uint)(base_byte >> 4u),
                    first_seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            uint second_low = 0u;
            uint second_high = 0u;
            if (has_second) {
                second_low = perturb_code(
                    (uint)(base_byte & 0x0fu),
                    second_seed,
                    leaf.element_offset + first,
                    leaf
                );
                if (first + 1u < leaf.length) {
                    second_high = perturb_code(
                        (uint)(base_byte >> 4u),
                        second_seed,
                        leaf.element_offset + first + 1u,
                        leaf
                    );
                }
            }
            for (uint h = 0u; h < params.history; ++h) {
                __global const uchar* observation =
                    rows + ((ulong)history_slots[h]) * ((ulong)params.row_bytes);
                uchar observed = observation[leaf.byte_offset + first_byte + local_byte];
                float first_low_delta =
                    ((float)first_low - (float)(observed & 0x0fu)) * leaf.scale;
                first_distances[h] = fma(
                    first_low_delta,
                    first_low_delta * leaf.weight,
                    first_distances[h]
                );
                if (first + 1u < leaf.length) {
                    float first_high_delta =
                        ((float)first_high - (float)(observed >> 4u)) * leaf.scale;
                    first_distances[h] = fma(
                        first_high_delta,
                        first_high_delta * leaf.weight,
                        first_distances[h]
                    );
                }
                if (has_second) {
                    float second_low_delta =
                        ((float)second_low - (float)(observed & 0x0fu)) * leaf.scale;
                    second_distances[h] = fma(
                        second_low_delta,
                        second_low_delta * leaf.weight,
                        second_distances[h]
                    );
                    if (first + 1u < leaf.length) {
                        float second_high_delta =
                            ((float)second_high - (float)(observed >> 4u)) * leaf.scale;
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
        for (uint element = tile.start + thread_index; element < end; element += THREADS) {
            uint first_value = perturb_code(
                (uint)base[leaf.byte_offset + element],
                first_seed,
                leaf.element_offset + element,
                leaf
            );
            uint second_value = has_second
                ? perturb_code(
                    (uint)base[leaf.byte_offset + element],
                    second_seed,
                    leaf.element_offset + element,
                    leaf
                )
                : 0u;
            for (uint h = 0u; h < params.history; ++h) {
                __global const uchar* observation =
                    rows + ((ulong)history_slots[h]) * ((ulong)params.row_bytes);
                float first_delta =
                    ((float)first_value - (float)observation[leaf.byte_offset + element])
                    * leaf.scale;
                first_distances[h] = fma(
                    first_delta,
                    first_delta * leaf.weight,
                    first_distances[h]
                );
                if (has_second) {
                    float second_delta =
                        ((float)second_value - (float)observation[leaf.byte_offset + element])
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

    __local float partials[THREADS];
    for (uint h = 0u; h < params.history; ++h) {
        partials[thread_index] = first_distances[h];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
            if (thread_index < stride) {
                partials[thread_index] += partials[thread_index + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (thread_index == 0u) {
            ulong offset =
                (((ulong)first_candidate) * ((ulong)params.history) + ((ulong)h))
                * ((ulong)params.tiles)
                + ((ulong)tile_index);
            partials_out[offset] = partials[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (has_second) {
            partials[thread_index] = second_distances[h];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = THREADS >> 1u; stride > 0u; stride >>= 1u) {
                if (thread_index < stride) {
                    partials[thread_index] += partials[thread_index + stride];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (thread_index == 0u) {
                ulong offset =
                    (((ulong)(first_candidate + 1u)) * ((ulong)params.history) + ((ulong)h))
                    * ((ulong)params.tiles)
                    + ((ulong)tile_index);
                partials_out[offset] = partials[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

__kernel void score_trials(
    __global const float* partials_in,
    __global const float* outcomes,
    __global const float* draws,
    __global float* scores,
    Params params
) {
    uint candidate_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (candidate_index >= params.candidates) {
        return;
    }
    __local float partials[THREADS];
    __local float nearest_distances[MAX_HISTORY];
    __local uint nearest_indices[MAX_HISTORY];
    if (thread_index == 0u) {
        for (uint k = 0u; k < params.neighbors; ++k) {
            nearest_distances[k] = INFINITY;
            nearest_indices[k] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint h = 0u; h < params.history; ++h) {
        float accum = 0.0f;
        ulong base =
            (((ulong)candidate_index) * ((ulong)params.history) + ((ulong)h))
            * ((ulong)params.tiles);
        for (uint tile = thread_index; tile < params.tiles; tile += THREADS) {
            accum += partials_in[base + (ulong)tile];
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
                if (
                    distance < nearest_distances[k]
                    || (distance == nearest_distances[k] && h < nearest_indices[k])
                ) {
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
                nearest_indices[insert_at] = h;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (thread_index == 0u) {
        float weight_sum = 0.0f;
        float weighted_value = 0.0f;
        for (uint k = 0u; k < params.neighbors; ++k) {
            float variance =
                1.0e-9f
                + params.epistemic_scale * nearest_distances[k]
                + params.aleatoric_scale;
            float weight = 1.0f / fmax(variance, 1.0e-12f);
            weight_sum += weight;
            weighted_value += weight * outcomes[nearest_indices[k]];
        }
        float mean = weighted_value / fmax(weight_sum, 1.0e-12f);
        float se = sqrt(1.0f / fmax(weight_sum, 1.0e-12f)) * params.y_scale;
        if (params.acquisition == 1u) {
            scores[candidate_index] = mean + se * draws[candidate_index];
        } else if (params.acquisition == 2u) {
            scores[candidate_index] = mean + se;
        } else {
            scores[candidate_index] = mean + params.beta * se;
        }
    }
}

__kernel void pick_trial(
    __global const float* scores,
    __global uint* choice,
    Params params
) {
    uint best = 0u;
    float best_score = scores[0];
    for (uint index = 1u; index < params.candidates; ++index) {
        float score = scores[index];
        if (score > best_score) {
            best = index;
            best_score = score;
        }
    }
    choice[0] = best;
}

__kernel void write_trial(
    __global uchar* rows,
    __global const Seed* seeds,
    __global const uint* choice,
    __global const Leaf* leaves,
    __global const Tile* tiles,
    Params params
) {
    uint tile_index = get_group_id(0);
    uint thread_index = get_local_id(0);
    if (tile_index >= params.tiles) {
        return;
    }
    Tile tile = tiles[tile_index];
    Leaf leaf = leaves[tile.leaf];
    Seed seed = seeds[choice[0]];
    __global const uchar* base =
        rows + ((ulong)params.base_slot) * ((ulong)params.row_bytes);
    __global uchar* trial =
        rows + ((ulong)params.trial_slot) * ((ulong)params.row_bytes);

    if (leaf.bits == 4u) {
        uint first_byte = tile.start / 2u;
        uint bytes = (tile.length + 1u) / 2u;
        for (uint local_byte = thread_index; local_byte < bytes; local_byte += THREADS) {
            uint first = tile.start + local_byte * 2u;
            uint low = perturb_code(
                code_at(base, leaf, first),
                seed,
                leaf.element_offset + first,
                leaf
            );
            uint high = 0u;
            if (first + 1u < leaf.length) {
                high = perturb_code(
                    code_at(base, leaf, first + 1u),
                    seed,
                    leaf.element_offset + first + 1u,
                    leaf
                );
            }
            trial[leaf.byte_offset + first_byte + local_byte] =
                (uchar)(low | (high << 4u));
        }
    } else {
        uint end = tile.start + tile.length;
        for (uint element = tile.start + thread_index; element < end; element += THREADS) {
            trial[leaf.byte_offset + element] = (uchar)perturb_code(
                code_at(base, leaf, element),
                seed,
                leaf.element_offset + element,
                leaf
            );
        }
    }
}
