# Metal and OpenCL ENNX

Status: design note

This document describes the next ENNX backend. The target is a Rust ENNX
optimizer that can keep its hot numerical path on an Apple Metal GPU or an
Intel GPU through OpenCL.

Keep the optimizer, ENN semantics, trust-region state, and `ask`/`tell`
contract in ENNX. Do not add another wrapper elsewhere.

## Why this work

The current experiments already run pure BO over approximately one
billion quantized model parameters. That demonstrates that the optimizer can
accept the search space. It does not mean that the ENN loop is GPU-resident.

The current hot path still has several possible host-side boundaries:

- observation history can be represented as host arrays;
- candidate generation can materialize large rows;
- KNN can return distances and indices to host code;
- posterior weights and uncertainty can be computed on the host;
- acquisition scoring and candidate selection can be computed on the host;
- trust-region updates can cause repeated synchronization;
- the model evaluator is a separate JAX/MPS path.

The target is a device-resident ENNX path. Rust remains responsible for
control flow, resource ownership, errors, and the public API. Metal and
OpenCL perform the large numerical operations.

## Current upstream state

The fork is `Kvutza/ennx`. Its configured upstream is
the prior implementation.

The upstream `main` branch recently introduced a substantial BPANN backend
refactor. It removed the previous experimental Metal and OpenCL files from
the ENNX crate. The fork must therefore not blindly merge upstream and lose
the accelerator work. The new backend should be built against the current
upstream backend and optimizer structure.

Before integrating upstream changes:

1. Save the current fork state and the local failure-tolerance changes.
2. Build and test the current fork.
3. Create an integration branch from the current upstream `main`.
4. Port the local fork changes explicitly.
5. Add the new accelerator backend on top of the new structure.

Do not restore the deleted `trials` and `weights` modules as a large copy.
Their useful behavior should be re-expressed through the current backend
interfaces.

## Definitions

Use these quantities consistently:

- `d`: dimension of one ENN input vector. In the LLM experiment, this can be
  approximately one billion.
- `n`: number of evaluated observations stored by ENN.
- `q`: number of query candidates in one search.
- `k`: number of neighbors used by the ENN posterior.
- `m`: number of candidates scored by the acquisition function.

BPANN mainly addresses very large `n`. It does not make one dense vector of
dimension `d` cheap. The Metal and OpenCL work must address the large `d`
calculation and the movement of candidate data.

## Goals

The backend should provide the following behavior:

1. Preserve the existing ENN and TuRBO semantics.
2. Keep the observation representation on the selected device.
3. Keep candidate generation and distance computation on the device when the
   device path is enabled.
4. Compute the ENN posterior without returning all distances to the host.
5. Compute UCB, Thompson sampling, and Pareto scores from the device
   posterior.
6. Support the same Rust optimizer through Metal and OpenCL.
7. Keep a CPU implementation as the exact reference.
8. Return only small results across the host boundary.
9. Make synchronization explicit and asynchronous where the backend permits
   it.
10. Provide parity tests that compare CPU, Metal, and OpenCL behavior.

## Non-goals for the first implementation

The first implementation should not attempt to:

- write an MLIR compiler or a new tensor compiler;
- rewrite the Granite or Olmoe model evaluator inside ENNX;
- port every old experimental kernel unchanged;
- make BPANN itself a Metal index;
- store one full billion-dimensional row for every possible candidate;
- support every distance metric before squared L2 is correct;
- optimize Thompson and Pareto before the UCB path is correct;
- remove the CPU reference implementation.

The model forward pass remains a separate evaluator boundary in the first
version. The ENNX optimizer can become GPU-resident before the model
evaluator is rewritten.

## Backend shape

The public optimizer continues to expose the existing Rust behavior. The
internal execution path should have three implementations:

```text
CPU reference
Metal execution
OpenCL execution
```

The common Rust layer owns:

- ENN posterior definitions;
- distance metric selection;
- neighbor ordering and tie-breaking;
- acquisition definitions;
- trust-region state transitions;
- observation identifiers;
- shape and error checks;
- command submission and lifetime management.

The device layer owns:

- device buffers;
- random-number generation used by candidate creation;
- tiled reductions;
- top-k selection;
- posterior arithmetic;
- acquisition arithmetic;
- asynchronous events and command buffers.

The CPU still submits work and handles scalar control flow. Fully GPU means
that the hot numerical path does not fall back to Python, NumPy, or large
host-side arrays.

## Host and device boundary

The current external `ask`/`tell` contract should remain usable. Internally,
the device path should add a way to represent a candidate without returning
its full vector.

The preferred candidate record is a deterministic descriptor:

```text
base model or incumbent id
candidate seed
layout id
trust-region parameters
leaf or block scales
candidate law version
```

The device can regenerate candidate values from this record. A full candidate
row should only be materialized in a device buffer when the model evaluator
requires it.

The minimum host result per proposal should be one of:

```text
candidate descriptor or device handle
selected candidate index
acquisition score
```

The minimum host result per evaluation should be:

```text
objective value and optional objective variance
```

If an existing caller still requires a host array, that should be an explicit
compatibility path, not the implementation used by the large experiment.

## Device state

The device state should be allocated once and reused across iterations.

It should contain, as applicable:

- observation descriptors or device rows;
- objective values and optional objective variances;
- the current incumbent descriptor;
- trust-region center and length;
- candidate seeds;
- query buffers;
- distance accumulators;
- top-k distances and indices;
- posterior mean and variance;
- acquisition scores;
- temporary buffers for append and rebuild operations.

The state must distinguish between:

1. a full vector representation;
2. a reproducible descriptor representation;
3. a low-dimensional sketch representation.

Using a sketch changes the metric unless it is only used as a coarse filter.
Every approximate representation must be labeled as such and compared with
the exact CPU reference.

## First metric

Implement squared L2 first:

```text
d2(x, y) = sum_j (x_j - y_j)^2
```

The reduction must be tiled over `d`. It must not allocate a `q x n x d`
array or a complete query-by-history distance matrix when only the nearest
`k` neighbors are needed.

Cosine distance can follow after the L2 path has parity tests. Mahalanobis
distance should not be added until the device representation of the metric
matrix and its memory cost are specified.

## Metal implementation

The Metal backend should use persistent `MTLBuffer` allocations and reuse
command queues and pipeline states.

The first kernels should be:

1. candidate descriptor or seed generation;
2. tiled squared-L2 distance;
3. local top-k selection;
4. top-k merge;
5. ENN posterior mean and variance;
6. UCB score and selected-index reduction.

The search should be hierarchical:

```text
dimension tile
  -> partial distance
  -> local top-k
  -> final top-k
  -> ENN posterior
  -> acquisition score
```

Do not synchronize after every kernel. Encode dependent operations in a
command buffer and wait only when the selected result or objective value is
needed.

For the current BO workload, `n` is small and `d` is large. The first Metal
implementation should therefore optimize the dimension reduction rather
than implement a complex approximate index.

## OpenCL implementation

OpenCL should implement the same operation contract, but it should not share
Metal source code. The common layer is the Rust operation contract and the
parity tests.

The OpenCL backend must account for:

- device and context creation;
- command queues and event dependencies;
- buffer alignment;
- supported integer and floating-point types;
- local-memory limits;
- work-group size;
- vendor-specific subgroup behavior;
- explicit synchronization.

The first OpenCL target is Intel GPU execution. A CPU OpenCL device should
also be usable for bring-up, but it must not be presented as GPU performance.

## ENN posterior and acquisition

After KNN returns `k` neighbors on the device, the backend should compute:

```text
local variance from distance
precision weights
posterior mean
posterior standard deviation
```

For UCB:

```text
score = mean + beta * standard_deviation
```

The backend should return only the best score and candidate index. The same
posterior output should feed Thompson sampling and Pareto scoring later.

Do not maintain separate Metal implementations of the ENN formulas for each
acquisition. Compute one posterior representation and apply acquisition
specific scoring to it.

## Candidate generation and trust region

The candidate law must be deterministic for a given seed, incumbent, layout,
and trust-region state.

Candidate generation should operate in blocks or leaves, so the backend can:

- generate a candidate tile;
- use it for distance or model evaluation;
- discard or reuse the tile;
- avoid a full host copy.

The trust-region update has two parts:

1. numerical scaling and clipping, which belong on the device when applied to
   candidate arrays;
2. the success/failure state transition, which can remain a small Rust control
   operation.

The effective-dimension setting for the trust-region failure clock must be
preserved when upstream changes are integrated.

## BPANN relationship

BPANN is the disk-backed approximate-neighbor backend for very large `n`.
It is not the Metal or OpenCL execution backend.

The intended long-term layout is:

```text
small or medium history: exact Metal/OpenCL search
large history: BPANN persistence and coarse retrieval
refinement: Metal/OpenCL distance and posterior scoring
```

Using descriptors or sketches can reduce storage, but it can also change the
ENN metric. This must be treated as an approximation and tested against an
exact reference.

## Implementation phases

### Phase 0: preserve and align

- Save the current fork and local changes.
- Integrate the current upstream backend structure on a separate branch.
- Reapply the local trust-region failure-tolerance change.
- Make the Rust and Python bindings compile.
- Run the existing CPU and disk BPANN tests.

Exit condition: upstream-aligned fork passes its existing tests and the
the current `turbo-enn` experiment still starts.

### Phase 1: exact Metal KNN

- Reintroduce `IndexDriver::Metal` through the current ENNX backend.
- Use tiled squared-L2 search.
- Keep history and query buffers on Metal after initial upload.
- Return only `k` distances and indices.
- Compare against Faiss/CPU on fixed random matrices.

Exit condition: neighbor indices match the CPU reference for supported test
sizes and the benchmark reports host bytes and kernel time.

### Phase 2: Metal posterior and UCB

- Compute ENN weights, mean, variance, and UCB on Metal.
- Avoid returning all distances to Rust or Python.
- Preserve deterministic tie-breaking.
- Compare posterior values and selected candidates against the CPU path.

Exit condition: CPU and Metal choose the same candidates on deterministic
fixtures within defined floating-point tolerances.

### Phase 3: Metal candidate path

- Generate candidates from descriptors and seeds on Metal.
- Apply trust-region bounds on Metal.
- Return a candidate handle or seed to the evaluator boundary.
- Add asynchronous append and update operations.

Exit condition: a BO run uses the Metal ENN path without materializing
the full candidate in Python.

### Phase 4: OpenCL parity

- Implement the same KNN and posterior operations in OpenCL.
- Run the same CPU parity fixtures.
- Test on an Intel GPU and a CPU OpenCL device.

Exit condition: the OpenCL backend passes the shared semantic tests.

### Phase 5: additional acquisition methods

- Thompson sampling from the device posterior.
- Pareto scoring for multiple objectives.
- Batch candidate selection.

Exit condition: acquisition-specific tests agree with the CPU reference on
the same posterior fixtures.

## Tests

Every backend needs tests for:

- empty and one-row history;
- `k = 1` and `k` larger than the history;
- tied distances;
- zero vectors and repeated rows;
- dimension tiles that do not divide `d`;
- candidate seeds and reproducibility;
- L2 and later cosine metrics;
- posterior mean and variance;
- UCB selection;
- append without full rebuild;
- device loss or unavailable feature handling.

The CPU implementation is the reference. Tests should compare values and
selected indices, not merely successful execution.

## Benchmarks

Measure at least:

```text
history n: 20, 1,000, 100,000
dimension d: 1K, 1M, 100M, 1B where hardware permits
queries q: 1, 4, 16
k: 1, 2, 10
metric: squared L2
```

Report:

- total search time;
- KNN kernel time;
- posterior and acquisition time;
- candidate generation time;
- host-to-device bytes;
- device-to-host bytes;
- peak device memory;
- CPU reference time;
- numerical disagreement;
- neighbor recall if an approximate representation is used.

The first useful performance result is not a claim that the whole BO loop is
under a particular time. It is a measured reduction in host traffic and a
reproducible comparison of CPU, Metal, and OpenCL search and scoring.

## Definition of done

The first complete backend is done when:

1. ENNX keeps the hot ENN state on Metal or OpenCL.
2. KNN, posterior, and UCB execute without NumPy arrays in the loop.
3. CPU, Metal, and OpenCL pass shared semantic tests.
4. The experiment uses the existing `ask`/`tell` surface.
5. The 1B-dimensional experiment can run through the new path.
6. Host transfer volume is measured and documented.
7. BPANN remains available as the large-history storage tier.

The model forward pass can be migrated later. It should not be mixed into the
first ENNX backend milestone.

## Questions for implementation

Before choosing a final device representation, answer these questions with
small benchmarks:

1. Does exact full-space L2 over the current quantized representation fit the
   available Metal memory and bandwidth?
2. Can a candidate be regenerated from a seed and layout without changing the
   current perturbation law?
3. Is f32 accumulation accurate enough at the target dimension, or is a
   blockwise compensated reduction required?
4. Does the Metal device support the required integer and quantized operations?
5. Which OpenCL types and subgroup operations are available on the Intel GPU?
6. Does a sketch preserve enough nearest-neighbor recall to retain useful ENN
   behavior?

The first answerable question is number 1. That should be the first benchmark
after Phase 0.
