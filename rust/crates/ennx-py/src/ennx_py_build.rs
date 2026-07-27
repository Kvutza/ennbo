#[path = "link_rpath.rs"]
mod link_rpath;

include!("ennx_py_build_api.inc.rs");
define_ennx_py_build_api!(link_rpath);
