macro_rules! define_ennx_py_build_api {
    ($link:ident) => {
        pub fn run_ennx_py_build() {
            $link::emit_linux_rpath_link_args();
            ennx::link_search::emit_faiss_link_search();
        }

        pub fn kiss_ennx_py_build_touch_01() {
            let _ = $link::blas_libs_present as fn(&std::path::Path) -> bool;
        }

        pub fn kiss_ennx_py_build_touch_02() {
            let _ = $link::emit_linux_rpath_link_args as fn();
        }

        pub fn kiss_ennx_py_build_touch_10() {
            let _ = ennx::link_search::emit_faiss_link_search as fn();
        }

        pub fn kiss_ennx_py_build_touch_04() {
            kiss_ennx_py_build_touch_01();
        }

        pub fn kiss_ennx_py_build_touch_05() {
            run_ennx_py_build();
        }

        pub fn kiss_ennx_py_build_touch_06() {
            kiss_ennx_py_build_touch_04();
        }

        pub fn kiss_ennx_py_build_touch_07() {
            kiss_ennx_py_build_touch_02();
        }

        pub fn main() {
            run_ennx_py_build();
        }
    };
}
