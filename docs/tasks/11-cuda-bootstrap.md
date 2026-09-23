# Task 11: CUDA C0/C1 bootstrap and ownership

Scope: implement C0/C1 from the CUDA plan. No additional model benchmark,
performance matrix, full evaluation, toolkit installation, or driver changes.
The CPU forward engine and CLI dispatch remain unchanged.

## GPU access diagnosis

The sandbox hides `/dev/dxg`: CUDA returns error 100 and NVML reports that GPU
access is blocked by the operating system. GPU commands require approved
execution outside the sandbox. Compilation and CPU tests run inside it.

Outside the sandbox `/dev/dxg` exists and `nvidia-smi` sees RTX 4070 Ti. However,
the default loader selects the installed Linux driver library instead of WSL's
library and CUDA still reports no device. Using `/usr/lib/wsl/lib` for this
process fixes execution; no system library/symlink/config was changed:

```sh
make smollm cuda build/cuda-lifecycle
LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_CACHE_PATH="$PWD/.cache/cuda" ./build/cuda-bootstrap
LD_LIBRARY_PATH=/usr/lib/wsl/lib make check-cuda
make check
```

This is a WSL-specific invocation, not a portable default for other machines.
Run GPU commands with host device access. Setting LD_LIBRARY_PATH alone does
not grant access through the sandbox. The runtime version is 11050, driver API
version 13010, cuBLAS version 11704; installed nvcc is 11.5.119. Code generation
uses sm_86 cubin plus compute_86 PTX, not unsupported sm_89.

## Implementation

- `include/sm_cuda.h`: pure C ABI, separate opaque device model/session. The
  existing `smollm.h` now has C++ linkage guards; host pointer APIs retain their
  meanings. No forward/trace functions are advertised before implementation.
- Model: preflight FP32 tensors through public validated-model accessors,
  copy config, upload all payloads once, own each device allocation. Embedding
  has one allocation for future embedding and tied output projection. Host
  model may be freed after creation.
- Session: borrow immutable device model; independently own nonblocking stream,
  cuBLAS handle, head-major FP32 K/V allocations and reusable scratch. Pedantic
  math and host scalar pointer mode are explicit. Frequency constants match
  CPU powf. Logits/staging are deferred until forward requires them.
- Checked memory arithmetic, invalid sizes/dtypes rejected before allocation,
  output pointers cleared on failure, partial construction cleanup. Device
  selection is scoped to the object and the caller's device is restored.
- Reset synchronizes and clears status/position. KV bytes are not zeroed:
  future attention must restrict visibility to the committed position. Actual
  old-token invisibility remains a C3 forward/cache gate, not a C1 claim.
- CPU library includes explicit unsupported stubs and has no CUDA dependency.
  `make cuda` builds `libsmollm-cuda.so` and the C-caller bootstrap. C sources
  use GCC; CUDA sources use nvcc; objects and outputs are separated. There is
  no CUDA CLI executable yet; `smollm-cuda` CLI belongs to C5.
- Failure injection is test-only, compiled into separate test objects. The
  production library has no failure-control global or environment hook.

## Verification scope

C0 uses a pure C caller, device allocation/copy, an add-one kernel, and a
non-square FP32 cuBLAS GEMM with independently written exact expected values.
The CUDA suite separately requires cubin execution with PTX disabled and PTX
execution with forced JIT. Requested GPU checks fail rather than skip if GPU
access is missing.

C1 tests compare every synthetic uploaded weight byte, release the host model,
create two independent sessions, verify scratch/KV accounting, frequency,
stream/handle modes, reset and canaries, and sweep every injected device
initialization failure. The production shared C ABI is also exercised through
ctypes. Synthetic inputs reuse the existing exporter; no model download or
real-model inference is involved.

CPU regression: `make check`: **144 passed, 4 CUDA tests skipped**, 8.68 seconds.
This includes the prior 143 tests plus the CPU stub contract. Existing suite
contains tiny fixture-based benchmark-command unit tests; no new performance
measurement campaign or real-model benchmark was run.

CUDA suite: **5 passed in 204.38 seconds**, including forced PTX JIT (first-run
compilation dominates this test duration; it is not an inference benchmark).
Rollback sweep covers 40 model and 11 session initialization failure points.
`LD_DEBUG=libs` confirms the default Linux driver path and the successful WSL
driver path. Versions, actual library paths and hashes are recorded in
`manifests/cuda-build.json`.

Compute Sanitizer 2021.3.1: **unavailable, not passed**. Both bootstrap and
lifecycle memcheck attempts first failed to find `libsanitizer-collection.so`.
With `--injection-path /usr/lib/nvidia-cuda-toolkit/compute-sanitizer`, both failed
with `Target application terminated before first instrumented API call` and
`couldn't find exit code` (exit 255). No tooling was installed or replaced.
Normal GPU tests pass, but this is not evidence of sanitizer cleanliness.
C2–C6 (operators, cache/attention, forward, CLI, numerical parity) remain pending.
