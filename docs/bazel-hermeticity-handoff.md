# Bazel hermeticity and Pixi consumer handoff

## Goal

Make Bazel the sole build, test, native dependency, toolchain, and wheel
packaging system for this ENNBO fork. Consumer projects will use Pixi only to
install a prebuilt, platform-specific wheel from a release URL.

The desired boundary is:

```text
pinned Bazel toolchains
        |
        v
Bazel build + Bazel artifact verification
        |
        v
platform wheel release assets
        |
        v
consumer pixi.toml
```

Do not introduce Maturin, an editable/source installation, a custom PEP 517
backend, a separate virtual environment, or a `bazel run` installer.

## Repository rules

- Use Jujutsu (`jj`) for all repository operations. Do not invoke `git`.
- Invoke Bazel directly. Do not wrap Bazel with `pixi run`.
- The working copy contains unrelated and overlapping user work. Do not restore,
  squash, split, commit, push, or publish it without first resolving exact
  ownership with the user.
- Use `apply_patch` for source edits.
- Preserve the BPANN/WeightSearch work and the existing multi-trust-region work.
- Run repository gates before any push or release.

Workspace:

```text
/Users/mehulbafna/Desktop/yubo/ennbo
```

Current Jujutsu state at handoff:

```text
Working copy: lttskrwq
Parent:       ysvnvpyz
Bookmark:     main
```

Always refresh this with:

```sh
jj status
jj diff --summary
```

## What is already good

- `.bazelversion` pins Bazel 9.2.0.
- `MODULE.bazel` pins `rules_rust`, `rules_python`, `rules_cc`,
  `rules_foreign_cc`, `platforms`, and OpenMP through Bzlmod.
- Rust 1.88.0 is registered through the `rules_rust` toolchain extension.
- Rust dependencies use checked-in Cargo and Crate Universe lockfiles.
- FAISS 1.12.0 and OpenBLAS 0.3.32 source archives have SHA-256 checksums.
- FAISS is compiled directly by Bazel; it does not discover Homebrew/Conda
  FAISS.
- The macOS wheel statically contains OpenMP and uses only Apple system
  frameworks/libraries at runtime.
- Bazel 9 uses a strict action environment by default. An audited Rust action
  received a static `/bin:/usr/bin:/usr/local/bin` `PATH` and did not inherit
  Pixi's `RUSTFLAGS`, `DYLD_LIBRARY_PATH`, or Conda library paths.
- The specialized BPANN, FAISS, and Metal Bazel targets pass.

## Release-blocking findings

### 1. The macOS wheel tag is false

The current wheel is named:

```text
ennbo-0.3.14-cp313-cp313-macosx_11_0_arm64.whl
```

Its `WHEEL` metadata contains:

```text
Tag: cp313-cp313-macosx_11_0_arm64
```

However, both C++ and Rust link actions currently receive:

```text
-mmacosx-version-min=26.1
```

The final Mach-O confirms:

```text
LC_BUILD_VERSION
  minos 26.1
  sdk   26.1
```

This wheel must not be released as macOS 11 compatible.

Cause: Bazel's `--macos_minimum_os` is unset, so the local Xcode SDK version is
used as the deployment target.

First fix to test:

```text
build:macos --macos_minimum_os=11.0
```

Also consider setting the release default for macOS host builds, because users
may build `//:python_wheel --config=release` without `--config=macos`.

Verify the fixed action graph:

```sh
bazel aquery 'mnemonic("Rustc", //rust/crates/enn-py:enn_rust_metal)' \
  --config=release --config=macos --output=textproto

bazel aquery 'mnemonic("CppCompile", @faiss_src//:faiss)' \
  --config=release --config=macos --output=textproto
```

Verify the built extension with `otool -l` and require `minos 11.0`.

### 2. Python 3.13 is metadata, not a toolchain invariant

`BUILD.bazel` hard-codes:

```text
abi = "cp313"
python_tag = "cp313"
python_requires = ">=3.13,<3.14"
```

No Python toolchain is registered in `MODULE.bazel`. `rules_python` therefore
chooses its default interpreter for wheel-making, while the output is manually
labelled CPython 3.13.

Register one Python version explicitly:

```starlark
python = use_extension(
    "@rules_python//python/extensions:python.bzl",
    "python",
)
python.defaults(python_version = "3.13")
python.toolchain(python_version = "3.13")
use_repo(python, "python_3_13")
```

The pinned `rules_python` documentation is available inside Bazel's external
repository under:

```text
external/rules_python+/docs/toolchains.md
```

Registration alone is insufficient. PyO3 must also be configured and tested
against CPython 3.13 rather than merely linked with unresolved Python symbols
on macOS.

Investigate the pinned `rules_rust` Crate Universe annotation support for:

```text
build_script_env
PYO3_CROSS
PYO3_CROSS_PYTHON_VERSION
PYO3_CONFIG_FILE
```

The interrupted investigation command was:

```sh
rg -n \
  "build_script_env|PYO3_CROSS_PYTHON_VERSION|PYO3_CONFIG_FILE" \
  <bazel-output-base>/external/rules_rust+ \
  <bazel-output-base>/external/rules_rust++crate+crates__pyo3-build-config-0.22.6
```

Do not claim CPython 3.13 compatibility until the built wheel is imported by a
Bazel test running the pinned 3.13 interpreter.

### 3. C/C++ is still host-discovered

Rust is downloaded and pinned. C/C++ currently goes through:

```text
external/rules_cc++cc_configure_extension+local_config_cc
```

That means compiler selection is local and is not fully hermetic. The Apple SDK
is necessarily a platform capability, but its version and deployment target
must be explicit in release CI. Linux and Windows should use pinned downloadable
C/C++ toolchains rather than `local_config_cc`.

Evaluate a Bzlmod-compatible pinned LLVM toolchain. Do not add a toolchain
without checksummed downloads and all three declared target platforms.

OpenBLAS also uses `rules_foreign_cc`. Confirm that its CMake and Ninja
toolchains are Bazel-provided and pinned rather than host-discovered. The root
module currently does not explicitly configure either tool.

### 4. The OS configs are currently cosmetic

`.bazelrc` contains:

```text
build:macos --define=ennbo_platform=macos
build:linux --define=ennbo_platform=linux
build:windows --define=ennbo_platform=windows
```

No BUILD target reads these defines. Platform selection actually uses host OS
constraints. Therefore `--config=macos`, `--config=linux`, and
`--config=windows` do not select real target platforms.

Choose one honest model:

1. Host-native releases only: remove the fake configs, clearly document that
   each wheel is built on its target OS, and pin each CI runner/toolchain.
2. Real cross-platform configs: define `platform()` targets and configure
   `--platforms`, execution platforms, C/C++, Rust, Python, SDK, and accelerator
   constraints correctly.

Host-native release jobs are likely the smaller and safer first implementation,
especially for Apple Metal and the Apple SDK.

## Artifact verification target

Add a Bazel test whose input is `//:python_wheel`. It must inspect the artifact
that will actually be released, not a separately assembled runfiles tree.

For macOS it must verify:

- exactly one native extension exists at `enn/enn_rust.so`;
- wheel filename and `WHEEL` tag agree;
- `METADATA` name, version, and `Requires-Python` agree with the Bazel target;
- Mach-O architecture is `arm64`;
- `LC_BUILD_VERSION minos` is no newer than the wheel platform tag;
- no Homebrew, Conda, workspace, or Bazel output paths appear as runtime
  dependencies;
- the only dynamic dependencies are approved Apple system libraries/frameworks;
- the wheel installs/imports under the pinned Bazel CPython 3.13 interpreter;
- `import enn` and `from enn import enn_rust` succeed;
- a small native API call succeeds, rather than testing import alone.

Use a Bazel `py_test` only for this release-artifact verification. Do not
reintroduce the previously removed design that mirrored `src/enn` plus the
extension into a runfiles-only development package.

Linux verification must eventually inspect ELF architecture, minimum glibc
policy, RPATH/RUNPATH, and shared dependencies. Windows verification must
inspect PE architecture and DLL dependencies.

Add this wheel audit to the canonical release gate, not necessarily every
fast unit-test invocation.

## Pixi boundary

The fork's root `pixi.toml` still contains build-era dependencies and settings:

- `git`
- `rust`
- `cmake`
- `ninja`
- `libfaiss`
- Linux `c-compiler` and `cxx-compiler`
- `CC` and `CXX`
- macOS `DYLD_LIBRARY_PATH`
- macOS `RUSTFLAGS`
- Bazel `check` and `wheel` tasks
- Cargo trial-test tasks

These must not define the Bazel build. Remove Bazel build tasks from Pixi.
Invoke Bazel directly:

```sh
bazel test ...
bazel build //:python_wheel --config=release ...
```

Decide separately whether the root Pixi environment remains for pure Python
source tests and documentation. If it remains, reduce it to those runtime/test
dependencies only.

The consumer contract is already represented by:

```text
examples/consumer/pixi.toml
```

It uses a direct platform wheel URL:

```toml
[target.osx-arm64.pypi-dependencies]
ennbo = { url = "https://github.com/Kvutza/ennbo/releases/download/v0.3.14/ennbo-0.3.14-cp313-cp313-macosx_11_0_arm64.whl" }
```

The URL will not resolve until a matching GitHub release asset is published.
Do not publish it until the code revision, wheel contents, wheel audit, and
release tag all match.

## Current gate results

The canonical Bazel gate was run through the old Pixi task before this audit:

```sh
pixi run -e ennbo bazel-test
```

Result:

- `//bazel/faiss:faiss_index_smoke`: passed
- `//rust/crates/bpann:bpann_test`: passed
- `//rust/crates/ennbo:trial_search_metal_test`: passed
- `//rust/crates/ennbo:ennbo_test`: timed out after 300 seconds

Within `ennbo_test`, all new BPANN/WeightSearch tests passed. The suite hung in:

```text
backend::flush_controller::tests::schedule_returns_while_soft_sync_in_flight
```

The test printed that it had run for more than 60 seconds and then consumed the
remaining suite timeout.

Before changing production synchronization, reproduce only this test with
direct Bazel:

```sh
bazel test //rust/crates/ennbo:ennbo_test \
  --test_filter=backend::flush_controller::tests::schedule_returns_while_soft_sync_in_flight \
  --test_output=streamed \
  --test_timeout=60
```

Inspect the test and controller implementation. Determine whether this is:

- a production deadlock;
- a test barrier that is never released;
- thread starvation caused by `--test-threads=1`;
- an incorrect interaction with the constrained test configuration.

Do not hide it by merely increasing the timeout or removing the test from the
gate.

The release wheel build was subsequently started but interrupted by the user
after it ran too long. Treat its result as unknown and rerun it directly after
the toolchain changes.

## Existing BPANN/Metal implementation to preserve

The current working copy contains a BPANN-to-WeightSearch integration:

- `rust/crates/ennbo/src/trials/bpann_history.rs`
- indexed shortlist APIs in `rust/crates/ennbo/src/trials.rs`
- streaming row replacement for CPU, Metal, and OpenCL
- Python bindings in `rust/crates/enn-py/src/py_weights.rs`
- Bazel Metal integration tests in
  `rust/crates/ennbo/tests/trial_search.rs`
- Python tests in `tests/test_weight_search.py`

Focused tests previously passed:

```sh
bazel test //rust/crates/ennbo:ennbo_test \
  --config=release \
  --config=constrained \
  --test_filter=trials:: \
  --test_timeout=60

bazel test //rust/crates/ennbo:trial_search_metal_test \
  --config=release \
  --config=constrained
```

The Bazel wheel previously built and imported successfully on the current
machine, but that does not override the false macOS minimum-version metadata
described above.

## Recommended execution order

1. Refresh `jj status`; record, but do not disturb, the mixed working copy.
2. Add explicit macOS minimum-version configuration.
3. Register the Bazel Python 3.13 toolchain.
4. Configure PyO3 consistently with that interpreter.
5. Build the macOS wheel directly with Bazel.
6. Add and pass the wheel artifact audit.
7. Reproduce and fix the flush-controller timeout.
8. Remove fake platform configs or replace them with real platforms.
9. Remove Bazel/build-tool responsibilities from root Pixi.
10. Run direct Bazel unit, Metal, FAISS, consumer-boundary, and wheel gates.
11. Inspect the final changes with `jj diff`; do not use Git.
12. Only after all gates pass, separate the intended change with `jj`, push it,
    and create a release whose revision matches the wheel.

## Definition of done

- Direct Bazel commands work outside Pixi.
- Mac wheel tag and Mach-O deployment target agree.
- Python ABI is controlled and tested by a pinned Bazel toolchain.
- No ambient package-manager compiler or library is required for the build
  beyond explicitly documented platform SDK/driver capabilities.
- Platform selection is honest.
- The hanging synchronization test is fixed, not bypassed.
- Release artifact verification is part of Bazel.
- The consumer Pixi manifest resolves a published, matching wheel.
- `jj status` and `jj diff` are used for repository verification.
- No release or push occurs while the gate is red.
