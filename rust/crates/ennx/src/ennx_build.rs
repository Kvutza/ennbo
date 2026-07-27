#[path = "link_search.rs"]
mod link_search;

fn main() {
    link_search::emit_faiss_link_search();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/faiss_bridge.cpp")
        .flag_if_supported("-std=c++17");
    if let Some(include_dir) = link_search::faiss_include_dir() {
        build.include(include_dir);
    }
    build.compile("enn_faiss_bridge");
}
