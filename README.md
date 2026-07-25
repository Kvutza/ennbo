# ENNBO

This repository is a public fork of
[yubo-research/enn](https://github.com/yubo-research/enn). Bazel is the
canonical native build and packaging system for the fork.

## Python

Build the wheel on the current platform:

```sh
bazel build //:python_wheel --config=release
python -m pip install bazel-bin/ennbo-*.whl
```

Python consumers depend on the resulting `ennbo` wheel and import `enn`.
They do not need Bazel, Cargo, FAISS headers, or a host package-manager FAISS
installation.

The checked-in wheel target currently emits the CPython 3.13 ABI. Release
automation must build additional ABI-tagged wheels for other supported Python
minor versions rather than relabeling one native binary.

The platform wheel selects the native accelerator automatically:

- macOS: FAISS/BPANN with Metal;
- Linux: FAISS/BPANN with OpenCL;
- Windows: FAISS/BPANN with OpenCL.

FAISS remains the CPU baseline and fallback on every platform. Bazel builds
FAISS from pinned source, uses the macOS Accelerate SDK, and builds pinned
OpenBLAS source for Linux and Windows. GPU driver runtimes are system
capabilities and are not bundled into the wheel.

## Bazel

Public native targets:

```text
//:rust_cpu
//:rust_metal
//:rust_opencl
//:rust_accelerators
//:python_extension
//:python_wheel
```

`//:rust_core` remains as a compatibility alias for `//:rust_cpu`.

Run the verified macOS build:

```sh
bazel test //:rust_tests //bazel/faiss:faiss_index_smoke --config=macos
bazel build //:python_wheel --config=macos --config=release
```

See [docs/bazel.md](docs/bazel.md) for platform and dependency details.
