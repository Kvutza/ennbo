use std::path::{Path, PathBuf};

fn blas_libs_present(dir: &Path) -> bool {
    ["libblas.so", "libopenblas.so", "libopenblas.so.0"]
        .iter()
        .any(|name| dir.join(name).exists())
}

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if !cfg!(target_os = "linux") {
        return;
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let lib = PathBuf::from(prefix).join("lib");
        if blas_libs_present(&lib) {
            let p = lib.to_string_lossy();
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{}", p);
        }
    }
    for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
        if blas_libs_present(Path::new(p)) {
            println!("cargo:rustc-cdylib-link-arg=-Wl,-rpath,{p}");
        }
    }
}
