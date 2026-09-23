# CUDA forward — C0/C1 완료 후 handoff

최신 요청: 추가 벤치마크/full 평가 없이 CUDA 구현. C0/C1 기반을 구현했으며
GPU forward 자체는 아직 없다. 다음 단계는 **C2 연산별 검증**이다.

## 읽을 문서

1. `AGENTS.md`
2. [CUDA 설계](design/cuda-forward.md)
3. [C0/C1 작업·GPU 접근 해결](tasks/11-cuda-bootstrap.md)
4. [CUDA build/검증 기록](../manifests/cuda-build.json)
5. [수치·형식 계약](design/contracts.md)

실제 branch/commit은 `git status --short --branch`, `git log -3 --oneline`으로
확인한다. 정상 작업 경로는 `/home/rmj/FPGA_LLM/llama2.c`다.

## GPU 실행에 필요한 환경

두 가지 문제가 확인됐다.

- 샌드박스 안에서는 `/dev/dxg`가 보이지 않아 GPU 실행이 불가능하다.
  GPU 테스트는 승인된 샌드박스 외부 실행을 사용한다.
- 외부에서도 기본 CUDA driver library 검색은 일반 Linux 설치를 선택해
  오류100을 반환했다. **프로세스에 `LD_LIBRARY_PATH=/usr/lib/wsl/lib`를 지정**하면
  WSL driver를 로드하며 kernel/cuBLAS 실행에 성공한다.

```sh
make smollm cuda build/cuda-lifecycle
LD_LIBRARY_PATH=/usr/lib/wsl/lib make check-cuda
make check
```

환경 변수는 프로세스 범위다. 전역 driver/toolkit/library 설정은 바꾸지 않았다.
RTX4070Ti(SM8.9), nvcc11.5.119를 그대로 사용하며 sm_86 cubin + compute_86 PTX를
생성한다. 강제 PTX 테스트는 cuBLAS JIT 때문에 첫 실행이 오래 걸릴 수 있다.

## 구현된 범위

- `include/sm_cuda.h`: 별도 opaque CUDA Model/Session C ABI.
- host model public accessor를 이용한 FP32 weight upload, config 복사, 부분 실패 정리.
- tied embedding은 한 allocation. 생성 후 host model을 해제할 수 있다.
- session별 stream/cuBLAS handle, FP32 KV/scratch/frequency/status.
- device 선택 복구, checked size 계산, Q8/KV8 거부, reset/free/getters.
- CPU-only stub, 선택적 CUDA shared library, 별도 test objects와 failure injection.
- C→CUDA kernel/cuBLAS fixture, weight byte 검증, 두 session 독립성,
  생성 실패 rollback, CPU 회귀 테스트. 실제 결과는 build manifest 참조.

C1은 **수명·메모리 기반**이다. reset 후 오래된 token의 비가시성은 position=0으로
무효화하는 계약까지만 구현됐으며 실제 attention에서의 검증은 C3에 남아 있다.
logits buffer/host staging, forward/decode/trace API와 CLI dispatch는 아직 없다.
CPU suite는 기존143개 + stub1개가 통과했고 GPU 전용4개는 별도 suite에서 실행한다.

## 다음 작업

- C2: embedding, row-major linear adapter, RMSNorm, split-half RoPE, SiLU/residual.
- C3: head-major KV scatter, GQA attention과 causal mask; cache/reset 독립 검증.
- C4: 전체 forward와 NONE/LAST/ALL, callback, 고정 HF 참조 수치 gate.
- C5: SmRun backend dispatch와 CLI. 현재 `build/smollm`은 여전히 CPU다.
- C6: 필요한 메모리 안전성 검사, 짧은/긴 고정 probe, 정리.

CPU `SmKernelOps`는 host pointer 전용이다. GPU device pointer를 넣지 않는다.
weights/KV/scratch는 GPU에 상주하고 layer마다 CPU로 activation을 가져오지 않는다.
FP32 logit `atol=1e-3, rtol=1e-4`, 평균 NLL 차이 `<=1e-4`를 유지한다.
실제 모델/토크나이저/참조는 기존 local artifact를 재사용한다.
추가 benchmark, full 평가, W8A8/KV8 CUDA, TF32/BF16 최적화는 현재 범위 밖이다.

기존 CPU 결과와 manifests/measurement-plan.json은 역사적 증거로 유지한다.
과거 CPU binary hash는 새 stub 포함 binary와 다르므로 과거 결과를 새 binary의
실측 결과로 쓰지 않는다. Skill/agent 설정은 프로젝트 로컬만 사용하며 `.codex`나
전역 설정을 바꾸지 않는다. 자동 위임 없이 이번 C0/C1은 단일 agent로 진행했다.
