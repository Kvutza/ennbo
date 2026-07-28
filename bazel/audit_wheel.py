import os
import subprocess
import sys
import tempfile
import zipfile


def audit_wheel(wheel_path: str):
    print(f"Auditing wheel: {wheel_path}")
    assert os.path.exists(wheel_path), f"Wheel not found at {wheel_path}"

    wheel_name = os.path.basename(wheel_path)
    assert "cp313-cp313-macosx_11_0_arm64" in wheel_name, (
        f"Unexpected wheel name: {wheel_name}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(wheel_path, "r") as z:
            z.extractall(tmpdir)

        dist_info = os.path.join(tmpdir, "ennx-0.0.0.dist-info")
        assert os.path.isfile(os.path.join(dist_info, "licenses", "LICENSE"))
        assert os.path.isfile(os.path.join(dist_info, "licenses", "NOTICE"))

        # 1. Verify exact single extension
        ext_path = os.path.join(tmpdir, "ennx", "ennx_rust.so")
        assert os.path.exists(ext_path), f"Missing extension at {ext_path}"

        # 2. Check Mach-O LC_BUILD_VERSION minos
        otool_out = subprocess.check_output(["otool", "-l", ext_path], text=True)
        assert "cmd LC_BUILD_VERSION" in otool_out, "LC_BUILD_VERSION missing"

        lines = otool_out.splitlines()
        minos_found = False
        for i, line in enumerate(lines):
            if "cmd LC_BUILD_VERSION" in line:
                block = "\n".join(lines[i : i + 6])
                assert "minos 11.0" in block, f"Expected minos 11.0, got:\n{block}"
                minos_found = True
                break
        assert minos_found, "Could not verify minos 11.0"

        # 3. Check dynamic shared libraries with otool -L
        otool_L = subprocess.check_output(["otool", "-L", ext_path], text=True)
        print("Dynamic libraries linked:\n", otool_L)
        forbidden_keywords = ["homebrew", "anaconda", "miniconda", "pixi", ".cache"]
        for line in otool_L.splitlines():
            line_lower = line.lower()
            for kw in forbidden_keywords:
                assert kw not in line_lower, (
                    f"Forbidden dependency path '{kw}' found in: {line}"
                )

        # 4. Test module import in CPython interpreter
        sys.path.insert(0, tmpdir)
        from ennx.ennx_rust import optimizer

        assert hasattr(optimizer, "WeightSearch"), (
            "Missing WeightSearch class on PyO3 optimizer module"
        )
        print(
            "Successfully imported ennx and verified native WeightSearch API from audited release wheel!"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: audit_wheel.py <path_to_wheel>")
        sys.exit(1)
    audit_wheel(sys.argv[1])
