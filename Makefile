.PHONY: all all-dynamic install install-dynamic clean test rust-test python-test lint \
	pypi-build pypi-publish pypi-auth-check

# Linux: conda/env often has no linkable libfaiss_c — use static Faiss for default Make targets.
# For dynamic Faiss (system libfaiss_c), use `make all-dynamic` / `install-dynamic` or plain maturin without --features.
MATURIN_STATIC_FAISS := $(shell test "$$(uname -s)" = Linux && echo --features static-faiss)

# Default: release extension (static Faiss on Linux so `make` works without libfaiss_c)
all:
	maturin build --release $(MATURIN_STATIC_FAISS)

# Same as `all` but dynamic Faiss — requires libfaiss_c at link time (e.g. faiss-devel / conda libfaiss_c)
all-dynamic:
	maturin build --release

# Install both the Rust extension and Python package
install:
	@echo "Building and installing Rust extension (see pyproject [tool.maturin])..."
	maturin develop --release $(MATURIN_STATIC_FAISS)
	@echo "Installing Python package (ennbo)..."
	pip install -e .
	@echo "Installation complete!"

install-dynamic:
	@echo "Building and installing Rust extension (dynamic Faiss; needs libfaiss_c)..."
	maturin develop --release
	@echo "Installing Python package (ennbo)..."
	pip install -e .
	@echo "Installation complete!"

# Run all tests (Rust and Python)
test: rust-test python-test

# Run Rust tests only
rust-test:
	cd rust && cargo test

# Run Python tests only
python-test:
	PYTHONPATH=src pytest -sv tests --tb=short

# Run linters
lint:
	cd rust && cargo clippy --all-targets --all-features -- -D warnings
	ruff check
	kiss check

# --- PyPI (ennbo): token in MATURIN_PYPI_TOKEN, or credentials in ~/.pypirc ---
# On Linux, static Faiss is required for typical publish hosts without libfaiss_c.
pypi-build:
	maturin build --release $(MATURIN_STATIC_FAISS)

# Note: `maturin publish` builds again before upload (same as a clean "build then publish").
pypi-publish:
	maturin publish --non-interactive $(MATURIN_STATIC_FAISS)

# Hits PyPI with your credentials but skips files already on the index (good auth smoke test).
pypi-auth-check: pypi-build
	maturin publish --non-interactive --skip-existing $(MATURIN_STATIC_FAISS)

# Clean build artifacts
clean:
	cd rust && cargo clean
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
