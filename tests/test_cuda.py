"""C0/C1 correctness only. Explicit check-cuda must fail if the GPU is absent."""
import ctypes as ct
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
GPU = pytest.mark.skipif(os.getenv("SM_TEST_CUDA") != "1", reason="run make check-cuda to require GPU")

class Error(ct.Structure):
    _fields_ = [("message", ct.c_char * 256)]


def library(name):
    lib = ct.CDLL(str(ROOT / "build" / name))
    ptr, err = ct.c_void_p, ct.POINTER(Error)
    lib.sm_model_load.argtypes = [ct.c_char_p, ct.POINTER(ptr), err]
    lib.sm_model_free.argtypes = [ptr]
    lib.sm_cuda_model_create.argtypes = [ptr, ct.c_int, ct.POINTER(ptr), err]
    lib.sm_cuda_model_free.argtypes = [ptr]
    lib.sm_cuda_session_create.argtypes = [ptr, ct.c_int, ct.c_size_t, ct.c_size_t, ct.POINTER(ptr), err]
    lib.sm_cuda_session_free.argtypes = [ptr]
    lib.sm_cuda_session_reset.argtypes = [ptr, err]
    for func in ("sm_cuda_model_weight_bytes", "sm_cuda_session_position", "sm_cuda_session_kv_bytes", "sm_cuda_session_scratch_bytes"):
        getattr(lib, func).argtypes = [ptr]
        getattr(lib, func).restype = ct.c_size_t
    return lib


def test_cpu_stub():
    lib = library("libsmollm.so")
    error, output = Error(), ct.c_void_p(1)
    assert lib.sm_cuda_model_create(None, 0, ct.byref(output), ct.byref(error)) == -1
    assert not output.value and b"CPU-only" in error.message
    output.value = 1
    assert lib.sm_cuda_session_create(None, 1, 8, 4, ct.byref(output), ct.byref(error)) == -1
    assert not output.value and b"CPU-only" in error.message
    assert lib.sm_cuda_session_reset(None, ct.byref(error)) == -1
    lib.sm_cuda_session_free(None)
    lib.sm_cuda_model_free(None)


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    import export_smollm
    folder = tmp_path_factory.mktemp("cuda-models")
    for dtype in ("f32", "q8"):
        export_smollm.export_synthetic(folder / f"{dtype}.bin", dtype)
    return folder


@GPU
@pytest.mark.parametrize("path", ["cubin", "ptx"])
def test_c_abi_kernel_cublas(path):
    env = dict(os.environ)
    env.pop("CUDA_FORCE_PTX_JIT", None)
    env.pop("CUDA_DISABLE_PTX_JIT", None)
    env["CUDA_DISABLE_PTX_JIT" if path == "cubin" else "CUDA_FORCE_PTX_JIT"] = "1"
    result = subprocess.run([ROOT / "build/cuda-bootstrap"], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"]


@GPU
def test_lifecycle_rollback_and_independence(tiny_models):
    result = subprocess.run([ROOT / "build/cuda-lifecycle", tiny_models / "f32.bin", tiny_models / "q8.bin"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["passed"]


@GPU
def test_production_c_abi_lifetime(tiny_models):
    lib = library("libsmollm-cuda.so")
    error = Error()
    host, model, a, b = (ct.c_void_p() for _ in range(4))
    try:
        assert lib.sm_model_load(str(tiny_models / "f32.bin").encode(), ct.byref(host), ct.byref(error)) == 0, error.message
        assert lib.sm_cuda_model_create(host, 0, ct.byref(model), ct.byref(error)) == 0, error.message
        lib.sm_model_free(host)
        host.value = None
        assert lib.sm_cuda_model_weight_bytes(model) > 0
        for session in (a, b):
            assert lib.sm_cuda_session_create(model, 1, 8, 4, ct.byref(session), ct.byref(error)) == 0, error.message
            assert lib.sm_cuda_session_position(session) == 0
            assert lib.sm_cuda_session_reset(session, ct.byref(error)) == 0, error.message
        lib.sm_cuda_session_free(a)
        a.value = None
        assert lib.sm_cuda_session_reset(b, ct.byref(error)) == 0, error.message
    finally:
        lib.sm_cuda_session_free(a)
        lib.sm_cuda_session_free(b)
        lib.sm_cuda_model_free(model)
        lib.sm_model_free(host)
