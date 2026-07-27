# ENNBO

Rust/Python Bayesian optimization with epistemic nearest-neighbor models.

## Build

```sh
bazel build //:wheel --config=release
```

The wheel is written to Bazel's output tree. It targets CPython 3.13.

## Test

```sh
bazel test //:check //:audit --config=release --config=constrained
```

Format Bazel files:

```sh
bazel run @buildifier_prebuilt//:buildifier -- -r .
```

## Targets

- `//:cpu` — CPU implementation
- `//:gpu` — platform GPU implementation
- `//:wheel` — Python wheel
- `//:check` — test suite
- `//:audit` — wheel audit

See [docs/bazel.md](docs/bazel.md) for dependency and platform details.
