load("@prelude//rust:cargo_package.bzl", "cargo")

_PROFILE = read_config("ennx", "profile", "dev")

_PROFILE_FLAGS = {
    "dev": [
        "-Copt-level=1",
        "-Cdebug-assertions=on",
        "-Coverflow-checks=on",
    ],
    "release": [
        "-Copt-level=3",
        "-Cdebug-assertions=off",
        "-Coverflow-checks=off",
    ],
}

if _PROFILE not in _PROFILE_FLAGS:
    fail("ennx.profile must be dev or release, got {}".format(_PROFILE))

def profile_rustc_flags():
    return _PROFILE_FLAGS[_PROFILE]

def app_rust_library(name, rustc_flags = None, **kwargs):
    cargo.rust_library(
        name = name,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        **kwargs
    )

def reindeer_rust_library(
        name,
        crate,
        rustc_flags = None,
        visibility = None,
        **kwargs):
    cargo.rust_library(
        name = name,
        crate = crate,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        visibility = visibility,
        **kwargs
    )

def reindeer_rust_binary(name, rustc_flags = None, **kwargs):
    cargo.rust_binary(
        name = name,
        rustc_flags = (rustc_flags or []) + profile_rustc_flags(),
        **kwargs
    )
