from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--openmp", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def _included(source: Path, root: Path) -> bool:
    relative = source.relative_to(root).as_posix()
    return not (
        relative.startswith(("faiss/cppcontrib/", "faiss/gpu/", "faiss/python/"))
    )


def main() -> None:
    args = _args()
    sources = [
        source
        for source in sorted((args.source / "faiss").rglob("*.cpp"))
        if _included(source, args.source)
    ]
    if not sources:
        raise SystemExit(f"no FAISS sources under {args.source}")

    packages = sorted(args.openmp.glob("pkg-llvm-openmp-*.tar.zst"))
    if len(packages) != 1:
        raise SystemExit(f"expected one llvm-openmp payload, found {packages}")
    openmp_root = args.out_dir / "openmp"
    openmp_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/bsdtar", "-xf", str(packages[0]), "-C", str(openmp_root)],
        check=True,
    )

    object_dir = args.out_dir / "faiss-objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "/usr/bin/clang++",
        "-c",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-mmacosx-version-min=11.0",
        "-Xpreprocessor",
        "-fopenmp",
        "-Wno-deprecated-declarations",
        "-Wno-unknown-pragmas",
        "-DFINTEGER=int",
        f"-I{args.source}",
        f"-I{openmp_root / 'include'}",
    ]

    def compile_one(source: Path) -> Path:
        digest = hashlib.sha256(
            source.relative_to(args.source).as_posix().encode()
        ).hexdigest()
        output = object_dir / f"{digest}.o"
        subprocess.run([*common, str(source), "-o", str(output)], check=True)
        return output

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        objects = list(executor.map(compile_one, sources))

    env = dict(os.environ)
    env["ZERO_AR_DATE"] = "1"
    archive = args.out_dir / "libfaiss.a"
    subprocess.run(
        ["/usr/bin/libtool", "-static", "-D", "-o", str(archive), *map(str, objects)],
        check=True,
        env=env,
    )
    shutil.copy2(openmp_root / "include" / "omp.h", args.out_dir / "omp.h")
    shutil.copy2(openmp_root / "lib" / "libomp.dylib", args.out_dir / "libomp.dylib")


if __name__ == "__main__":
    main()
