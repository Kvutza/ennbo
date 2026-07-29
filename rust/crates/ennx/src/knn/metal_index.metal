#include <metal_stdlib>
using namespace metal;

constant uint kThreads = 256;
constant uint kTileRows = 1024;
constant uint kMergeRows = 2048;

inline uint merge_width(uint k) {
    uint width = 1;
    while (width < 2 * k) {
        width <<= 1;
    }
    return width;
}

struct Params {
    uint rows;
    uint dim;
    uint queries;
    uint tile_start;
    uint tile_rows;
    uint k;
};

kernel void init_results(
    device float* result_distances [[buffer(0)]],
    device uint* result_indices [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * params.k;
    if (gid >= total) return;
    result_distances[gid] = INFINITY;
    result_indices[gid] = 0xffffffffu;
}

inline bool before(float distance_a, uint index_a, float distance_b, uint index_b) {
    return distance_a < distance_b
        || (distance_a == distance_b && index_a < index_b);
}

inline void order_pair(threadgroup float* distances, threadgroup uint* indices,
                       uint left, uint right, bool ascending) {
    float left_distance = distances[left];
    uint left_index = indices[left];
    float right_distance = distances[right];
    uint right_index = indices[right];
    bool swap_pair = ascending
        ? before(right_distance, right_index, left_distance, left_index)
        : before(left_distance, left_index, right_distance, right_index);
    if (swap_pair) {
        distances[left] = right_distance;
        indices[left] = right_index;
        distances[right] = left_distance;
        indices[right] = left_index;
    }
}

kernel void distance_rows(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float sum = 0.0f;
    for (uint d = 0; d < params.dim; ++d) {
        float delta = row[d] - query[d];
        sum = fma(delta, delta, sum);
    }
    distances[gid] = sum;
}

kernel void distance_rows_2(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float2 sum = 0.0f;
    uint d = 0;
    for (; d + 1 < params.dim; d += 2) {
        float2 delta = float2(row[d], row[d + 1]) - float2(query[d], query[d + 1]);
        sum = fma(delta, delta, sum);
    }
    float result = sum.x + sum.y;
    if (d < params.dim) {
        float delta = row[d] - query[d];
        result = fma(delta, delta, result);
    }
    distances[gid] = result;
}

kernel void distance_rows_4(
    device const float* rows [[buffer(0)]],
    device const float* queries [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    uint total = params.queries * kTileRows;
    if (gid >= total) return;
    uint query_index = gid / kTileRows;
    uint tile_index = gid - query_index * kTileRows;
    if (tile_index >= params.tile_rows) {
        distances[gid] = INFINITY;
        return;
    }
    uint row_index = params.tile_start + tile_index;
    device const float* row = rows + ulong(row_index) * ulong(params.dim);
    device const float* query = queries + ulong(query_index) * ulong(params.dim);
    float4 sum = 0.0f;
    uint d = 0;
    for (; d + 3 < params.dim; d += 4) {
        float4 delta = float4(row[d], row[d + 1], row[d + 2], row[d + 3])
                     - float4(query[d], query[d + 1], query[d + 2], query[d + 3]);
        sum = fma(delta, delta, sum);
    }
    float result = (sum.x + sum.y) + (sum.z + sum.w);
    for (; d < params.dim; ++d) {
        float delta = row[d] - query[d];
        result = fma(delta, delta, result);
    }
    distances[gid] = result;
}

kernel void local_topk(
    device const float* distances [[buffer(0)]],
    device float* output_distances [[buffer(1)]],
    device uint* output_indices [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query_index [[threadgroup_position_in_grid]]) {
    threadgroup float values[kTileRows];
    threadgroup uint indices[kTileRows];
    for (uint i = tid; i < kTileRows; i += kThreads) {
        values[i] = i < params.tile_rows
            ? distances[query_index * kTileRows + i]
            : INFINITY;
        indices[i] = i < params.tile_rows ? params.tile_start + i : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= kTileRows; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < kTileRows; i += kThreads) {
                uint partner = i ^ stride;
                if (partner > i) {
                    order_pair(values, indices, i, partner, (i & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < params.k; i += kThreads) {
        output_distances[query_index * params.k + i] = values[i];
        output_indices[query_index * params.k + i] = indices[i];
    }
}

kernel void merge_topk(
    device float* result_distances [[buffer(0)]],
    device uint* result_indices [[buffer(1)]],
    device const float* local_distances [[buffer(2)]],
    device const uint* local_indices [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint query_index [[threadgroup_position_in_grid]]) {
    threadgroup float values[kMergeRows];
    threadgroup uint indices[kMergeRows];
    uint width = merge_width(params.k);
    for (uint i = tid; i < width; i += kThreads) {
        if (i < params.k) {
            values[i] = result_distances[query_index * params.k + i];
            indices[i] = result_indices[query_index * params.k + i];
        } else if (i < 2 * params.k) {
            uint local = i - params.k;
            values[i] = local_distances[query_index * params.k + local];
            indices[i] = local_indices[query_index * params.k + local];
        } else {
            values[i] = INFINITY;
            indices[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= width; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < width; i += kThreads) {
                uint partner = i ^ stride;
                if (partner > i) {
                    order_pair(values, indices, i, partner, (i & size) == 0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < params.k; i += kThreads) {
        result_distances[query_index * params.k + i] = values[i];
        result_indices[query_index * params.k + i] = indices[i];
    }
}
