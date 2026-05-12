#!/usr/bin/env bash
set -eu

min_maj=3
min_min=24
ver=3.28.4

cmake_ok() {
  local c="$1" line maj mi rest
  [ -x "$c" ] || return 1
  line="$("$c" --version 2>/dev/null | head -n1 || true)"
  case "$line" in
    cmake\ version\ *)
      rest="${line#cmake version }"
      maj="${rest%%.*}"
      rest="${rest#*.}"
      mi="${rest%%.*}"
      if [ "$maj" -gt "$min_maj" ]; then return 0; fi
      if [ "$maj" -eq "$min_maj" ] && [ "$mi" -ge "$min_min" ]; then return 0; fi
      ;;
  esac
  return 1
}

try_exec() {
  local c
  for c in \
    /usr/bin/cmake \
    /usr/local/bin/cmake \
    /opt/homebrew/bin/cmake \
    /home/linuxbrew/.linuxbrew/bin/cmake; do
    if cmake_ok "$c"; then
      exec "$c" "$@"
    fi
  done
}

try_exec "$@"

os=$(uname -s)
arch=$(uname -m)
cache="${XDG_CACHE_HOME:-$HOME/.cache}/enn-cmake"
mkdir -p "$cache"

case "$os.$arch" in
  Linux.x86_64)
    dist="cmake-${ver}-linux-x86_64.tar.gz"
    want_sha=1f74731c80cbba3263c64fca6f6af0fb8dd1d06365425e404f79564773080d11
    sub="cmake-${ver}-linux-x86_64"
    cm="$cache/$sub/bin/cmake"
    url="https://github.com/Kitware/CMake/releases/download/v${ver}/${dist}"
    ;;
  Linux.aarch64)
    dist="cmake-${ver}-linux-aarch64.tar.gz"
    want_sha=74edb3d6f7d179dc5021bd9f3c4ac59c72bb2c4e6bea9abd8297d8ce0a385228
    sub="cmake-${ver}-linux-aarch64"
    cm="$cache/$sub/bin/cmake"
    url="https://github.com/Kitware/CMake/releases/download/v${ver}/${dist}"
    ;;
  Darwin.arm64|Darwin.x86_64)
    dist="cmake-${ver}-macos-universal.tar.gz"
    want_sha=ad47a7e8e3da180b7cff69efe337f4285305052a77f28960cd40ca66f2f5c894
    sub="cmake-${ver}-macos-universal"
    cm="$cache/$sub/CMake.app/Contents/bin/cmake"
    url="https://github.com/Kitware/CMake/releases/download/v${ver}/${dist}"
    ;;
  *)
    echo "enn_cmake.sh: no suitable system cmake and no binary for ${os}/${arch}" >&2
    exit 1
    ;;
esac

if [ -x "$cm" ]; then
  exec "$cm" "$@"
fi

tmp="$cache/${dist}.part"
curl -fLsS --retry 3 -o "$tmp" "$url"
if command -v sha256sum >/dev/null 2>&1; then
  echo "$want_sha  $tmp" | sha256sum -c -
else
  got=$(shasum -a 256 "$tmp" | awk '{print $1}')
  if [ "$got" != "$want_sha" ]; then
    echo "enn_cmake.sh: sha256 mismatch for $dist" >&2
    exit 1
  fi
fi
tar -xzf "$tmp" -C "$cache"
rm -f "$tmp"
exec "$cm" "$@"
