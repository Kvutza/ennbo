use std::path::{Path, PathBuf};

fn has_blas_for_link(dir: &Path) -> bool {
    dir.join("libblas.so").exists()
        || dir.join("libopenblas.so").exists()
        || dir.join("libopenblas.so.0").exists()
}

/// `faiss-sys` links `-lblas -llapack`; ensure the linker sees them.
fn emit_blas_lapack_link_search_linux() {
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if has_blas_for_link(&lib) {
            println!(
                "cargo:rustc-link-search=native={}",
                lib.to_str().expect("utf-8 conda lib path")
            );
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if has_blas_for_link(Path::new(p)) {
            println!("cargo:rustc-link-search=native={p}");
        }
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if cfg!(target_os = "linux") {
        emit_blas_lapack_link_search_linux();
    }
}
