def python_wheel(
        name,
        extension,
        extension_suffix,
        license,
        notice,
        package,
        platform_tag,
        python_srcs,
        runtime_libraries,
        target_compatible_with,
        version):
    """Builds a deterministic CPython 3.13 wheel around a PyO3 shared library."""
    filename = "{}-{}-cp313-cp313-{}.whl".format(package, version, platform_tag)
    native.genrule(
        name = name,
        srcs = python_srcs + runtime_libraries + [extension, license, notice],
        out = filename,
        cmd = " ".join([
            "$(exe //buck2/wheel:pack)",
            "--src-dir",
            "$SRCDIR",
            "--out",
            "$OUT",
            "--package",
            package,
            "--version",
            version,
            "--platform-tag",
            platform_tag,
            "--extension-suffix",
            extension_suffix,
        ]),
        target_compatible_with = target_compatible_with,
        visibility = ["PUBLIC"],
    )
