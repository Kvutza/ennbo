.PHONY: all clean test install rust-test python-test lint pypi-build pypi-publish pypi-auth-check

# Linux PyPI / minimal images: static Faiss (no system libfaiss_c). Local `make all` / `install` stay dynamic.
MATURIN_STATIC_FAISS := $(shell test "$$(uname -s)" = Linux && echo --features static-faiss)

# Default target: build the Rust extension in release mode (dynamic Faiss when libfaiss_c is available)
all:
	maturin build --release

# Install both the Rust extension and Python package
install:
	@echo "Building and installing Rust extension (see pyproject [tool.maturin])..."
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
