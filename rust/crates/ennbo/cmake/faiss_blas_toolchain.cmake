# Used as CMAKE_TOOLCHAIN_FILE so Faiss's `find_package(BLAS)` succeeds when
# building static `faiss-sys` on Linux.
cmake_minimum_required(VERSION 3.16)

set(_enn_blas "")
if(NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  foreach(
    _c IN ITEMS "$ENV{CONDA_PREFIX}/lib/libopenblas.so"
                "$ENV{CONDA_PREFIX}/lib/libopenblas.so.0"
                "$ENV{CONDA_PREFIX}/lib/libblas.so")
    if(EXISTS "${_c}")
      set(_enn_blas "${_c}")
      break()
    endif()
  endforeach()
endif()
if(_enn_blas STREQUAL "")
  foreach(
    _c IN ITEMS "/usr/lib/x86_64-linux-gnu/libopenblas.so"
                "/usr/lib/x86_64-linux-gnu/libopenblas.so.0"
                "/usr/lib/aarch64-linux-gnu/libopenblas.so"
                "/usr/lib/aarch64-linux-gnu/libopenblas.so.0")
    if(EXISTS "${_c}")
      set(_enn_blas "${_c}")
      break()
    endif()
  endforeach()
endif()

if(NOT _enn_blas STREQUAL "")
  set(BLAS_LIBRARIES
      "${_enn_blas}"
      CACHE FILEPATH "BLAS for Faiss (enn toolchain)" FORCE)
  set(LAPACK_LIBRARIES
      "${_enn_blas}"
      CACHE FILEPATH "LAPACK for Faiss (enn toolchain)" FORCE)
endif()
