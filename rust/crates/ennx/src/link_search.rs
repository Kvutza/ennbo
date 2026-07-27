use std::path::{Path, PathBuf};

pub fn emit_link_search(p: &str) {
    println!("cargo:rustc-link-search=native={p}");
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
    }
    println!("cargo:rustc-link-lib=dylib=faiss");
}

pub fn has_faiss(dir: &Path) -> bool {
    dir.join("libfaiss.dylib").exists()
        || dir.join("libfaiss.so").exists()
        || dir.join("libfaiss.dll.a").exists()
        || dir.join("faiss.lib").exists()
}

pub fn faiss_include_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(dir) = std::env::var("FAISS_INCLUDE_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(dir) = std::env::var("FAISS_LIB_DIR") {
        let lib = PathBuf::from(dir);
        candidates.push(lib.join("../include"));
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        candidates.push(if cfg!(target_os = "windows") {
            prefix.join("Library/include")
        } else {
            prefix.join("include")
        });
    }
    if cfg!(target_os = "macos") {
        candidates.extend([
            PathBuf::from("/opt/homebrew/opt/faiss/include"),
            PathBuf::from("/usr/local/opt/faiss/include"),
        ]);
    }
    if cfg!(target_os = "linux") {
        candidates.extend([
            PathBuf::from("/usr/include"),
            PathBuf::from("/usr/local/include"),
        ]);
    }
    candidates
        .into_iter()
        .find(|dir| dir.join("faiss/IndexFlat.h").exists())
}

pub fn emit_faiss_link_search() {
    println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=FAISS_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    if let Ok(dir) = std::env::var("FAISS_LIB_DIR") {
        let path = Path::new(&dir);
        if has_faiss(path) {
            emit_link_search(&dir);
            return;
        }
        panic!("FAISS_LIB_DIR does not contain a Faiss library: {dir}");
    }
    if let Ok(prefix) = std::env::var("CONDA_PREFIX") {
        let prefix = PathBuf::from(prefix);
        let lib = if cfg!(target_os = "windows") {
            prefix.join("Library/lib")
        } else {
            prefix.join("lib")
        };
        if has_faiss(&lib) {
            emit_link_search(lib.to_str().expect("utf-8 conda lib path"));
            return;
        }
    }
    if cfg!(target_os = "macos") {
        for p in ["/opt/homebrew/opt/faiss/lib", "/usr/local/opt/faiss/lib"] {
            if has_faiss(Path::new(p)) {
                emit_link_search(p);
                return;
            }
        }
    }
    if cfg!(target_os = "linux") {
        for p in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"] {
            if has_faiss(Path::new(p)) {
                emit_link_search(p);
                return;
            }
        }
    }
    panic!(
        "Faiss was not found; install the Pixi libfaiss package or set FAISS_LIB_DIR and FAISS_INCLUDE_DIR"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn has_faiss_empty_dir() {
        let dir = std::env::temp_dir().join(format!("enn_faiss_check_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!has_faiss(Path::new(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn emit_faiss_link_search_smoke() {
        emit_faiss_link_search();
    }
}
