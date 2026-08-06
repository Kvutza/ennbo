# Testing `ennx-py`

`ennx-py` is a Python extension crate (PyO3 `cdylib`).
The reliable test path is Python-side after installing the extension.

## Why not `cargo test -p ennx-py` for wrapper behavior?

On some systems, Rust test binaries for PyO3 crates fail to link Python C symbols.
That is an embedding/link mode issue, not a missing algorithm implementation.

## Recommended workflow

From repo root:

1. Run source-only config tests:

```bash
cd /path/to/repo
pixi run -e ennx test
```

2. Build the wheel and run its isolated wheel smoke and API tests:

```bash
cd /path/to/repo
pixi run -e ennx buck2-verify
```

Do not combine `PYTHONPATH=src` with an extension installed only in
`site-packages`; the source package would shadow the installed wheel.

## Rust-side checks that should still pass

```bash
cd /path/to/repo/rust
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
