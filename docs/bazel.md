# Bazel build and distribution

Bazel owns native dependency resolution, Rust/C++ compilation, tests, and
Python wheel creation. A downstream Python project consumes the wheel; a
downstream Bazel project consumes the public module targets.

## Public targets

| Target | Contents |
| --- | --- |
| `//:rust_cpu` | Rust ENNBO with FAISS and BPANN |
| `//:rust_metal` | CPU target plus Metal kernels; macOS only |
| `//:rust_opencl` | CPU target plus OpenCL kernels |
| `//:rust_accelerators` | Host-selected Metal or OpenCL compatibility alias |
| `//:python_extension` | Host-selected native Python extension |
| `//:python_wheel` | Python sources plus the native extension |

`//:rust_core` remains a compatibility alias for `//:rust_cpu`.

CPU, Metal, and OpenCL are separate Bazel targets. Metal and OpenCL are no
longer forced into the same Rust feature closure.

## Native dependency contract

FAISS is present in every native target. Accelerator drivers augment the
index layer; they do not replace the CPU FAISS capability.

Bazel fetches checksum-pinned FAISS 1.12.0 source and compiles it directly.
The Rust graph does not run the legacy `faiss-sys` CMake build and does not
search Homebrew or another host package manager for FAISS.

On macOS, Bazel uses:

- the pinned LLVM OpenMP module;
- the Apple SDK Accelerate framework for BLAS/LAPACK;
- the system Metal framework for the host-selected wheel.

The explicit `//:rust_opencl` target remains available for OpenCL development
on macOS, while the default macOS wheel contains Metal rather than both GPU
stacks.

Linux and Windows select OpenCL rather than Metal. Bazel also fetches
checksum-pinned OpenBLAS 0.3.32 source and builds a static library for those
platforms. The build does not fall back to an ambient `-lopenblas`.

OpenBLAS uses its upstream CMake build through Bazel-managed
`rules_foreign_cc` toolchains. The source, CMake, and Ninja inputs are resolved
by Bazel rather than discovered through a user package manager.

## Dependency locking

`Cargo.Bazel.lock` and `Cargo.Accelerators.Bazel.lock` are Crate Universe
generator lockfiles. They are checked in deliberately: Bzlmod dependency
modules cannot repin repositories inside a consumer's read-only module cache.

The lockfiles cover:

- `aarch64-apple-darwin`;
- `x86_64-unknown-linux-gnu`;
- `x86_64-pc-windows-msvc`.

When Cargo manifests, supported triples, or crate annotations change, repin
from the ENNBO repository root:

```sh
CARGO_BAZEL_REPIN=1 bazel build //:rust_cpu
```

Consumers must never patch or regenerate ENNBO's lockfiles.

## Build and test

macOS:

```sh
bazel test //:rust_tests //bazel/faiss:faiss_index_smoke --config=macos
bazel build //:python_wheel --config=macos --config=release
```

Laptop-friendly resource limits are opt-in:

```sh
bazel build //:python_wheel --config=macos --config=release --config=constrained
```

The wheel appears under `bazel-bin/` and installs normally:

```sh
python -m venv .venv
.venv/bin/python -m pip install bazel-bin/ennbo-*.whl
.venv/bin/python -c "import enn, enn.enn_rust"
```

The local wheel target is tagged for CPython 3.13. Other Python minor versions
require separately compiled wheels; the native extension is not falsely marked
as a stable-ABI wheel.

No user-specific path is part of the build or install contract.

## Consumer checks

The repository contains two owner-boundary smoke fixtures:

- `tests/bazel_consumer` consumes ENNBO as a non-root Bzlmod module;
- `tests/python_consumer/smoke.py` imports an installed wheel.

They exist to catch root-only Crate Universe behavior and malformed wheel
layouts before a release is published.
