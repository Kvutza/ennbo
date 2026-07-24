# Bazel build

This overlay builds the ENNBO Rust workspace at `cd4c7dc` without changing its
Rust or Python sources. Bazel owns dependency resolution, compilation, tests,
and the Python extension artifact in this isolated JJ workspace.

## Native dependency contract

FAISS is part of every ENNBO build, including Metal and OpenCL builds. The
accelerator drivers augment the index layer; they do not replace the CPU
FAISS capability.

Bazel fetches the checksum-pinned FAISS 1.12.0 source release, matching
`faiss-sys` 0.7.0, and compiles both `faiss` and `faiss_c` directly. The
`faiss-sys` Cargo build script is disabled and its Rust library receives the
Bazel C API target through `link_deps`.

On macOS:

- Bazel's pinned LLVM OpenMP target supplies the OpenMP headers and runtime;
- the Apple SDK's Accelerate framework supplies BLAS/LAPACK;
- FAISS and OpenMP are linked statically into the extension;
- Metal and OpenCL ENNBO features are both enabled.

No external FAISS installation or native-library search environment is used.

Build and test the Rust crates:

```sh
bazel test //:rust_tests --config=macos
bazel test //bazel/faiss:faiss_index_smoke --config=macos
```

Build the optimized Python extension:

```sh
bazel build //:python_extension --config=macos --config=release
```

The resulting local artifact is:

```text
bazel-bin/rust/crates/enn-py/enn_rust.so
```

It can be loaded directly from the existing Pixi Python environment:

```sh
/Users/mehulbafna/Desktop/yubo/ennbo/.pixi/envs/ennbo/bin/python
```

`otool -L bazel-bin/rust/crates/enn-py/enn_rust.so` should show only Apple
SDK/system libraries, including Metal and Accelerate. It must not contain an
external FAISS or OpenMP dynamic-library path.

## Platform layout

The pinned FAISS source and Bazel C/C++ targets are shared by every platform.
Platform constraints select only the native runtime pieces:

| Platform | ENNBO backends | BLAS/LAPACK | OpenMP |
| --- | --- | --- | --- |
| macOS | FAISS, BPANN, Metal, OpenCL | Accelerate SDK | Bazel LLVM OpenMP |
| Linux | FAISS, BPANN, OpenCL | pinned OpenBLAS target | Bazel LLVM OpenMP |
| Windows | FAISS, BPANN, OpenCL | pinned OpenBLAS target | selected Windows toolchain runtime |

The FAISS target exposes an overridable `@faiss_src//:blas` label. The macOS
default is complete. Linux and Windows must override that label with the
checked-in pinned OpenBLAS target when those Rust toolchain lanes are enabled;
they must not fall back to an ambient `-lopenblas`.

The Bazel module graph is restricted to `aarch64-apple-darwin` so Cargo's
Linux-only static-FAISS feature does not leak into the verified macOS lane.
Linux and Windows get separate crate-universe/toolchain lanes rather than
unifying target-specific Cargo features into the macOS dependency graph.
