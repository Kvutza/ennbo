use std::path::{Path, PathBuf};

fn has_faiss(dir: &Path) -> bool {
    let faiss = ["libfaiss.dylib", "libfaiss.so", "faiss.lib"]
        .iter()
        .any(|name| dir.join(name).exists());
    let faiss_c = ["libfaiss_c.dylib", "libfaiss_c.so", "faiss_c.lib"]
        .iter()
        .any(|name| dir.join(name).exists());
    faiss && faiss_c
}

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("FAISS_LIB_DIR") {
        paths.push(path.into());
    }
    if let Some(prefix) = std::env::var_os("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        paths.push(if cfg!(target_os = "windows") {
            prefix.join("Library/lib")
        } else {
            prefix.join("lib")
        });
    }
    if cfg!(target_os = "macos") {
        paths.extend([
            PathBuf::from("/opt/homebrew/opt/faiss/lib"),
            PathBuf::from("/usr/local/opt/faiss/lib"),
        ]);
    } else if cfg!(target_os = "linux") {
        paths.extend([
            PathBuf::from("/usr/lib/x86_64-linux-gnu"),
            PathBuf::from("/usr/lib/aarch64-linux-gnu"),
            PathBuf::from("/usr/local/lib"),
        ]);
    }
    paths
}

fn main() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if std::env::var_os("CARGO_FEATURE_FAISS").is_none() || std::env::var_os("DOCS_RS").is_some() {
        return;
    }
    let lib = candidates()
        .into_iter()
        .find(|candidate| has_faiss(candidate))
        .expect("Faiss C and C++ libraries were not found; set FAISS_LIB_DIR");
    println!("cargo:rustc-link-search=native={}", lib.display());
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
