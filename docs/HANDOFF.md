# CUDA forward 전환 — 다음 세션 handoff

2026-09-23 최신 지시: **추가 성능 측정/full 평가는 중단하고 CUDA 작업으로 전환**한다.
이번 세션은 상세 계획만 작성하고 종료하라는 요청에 따라 **CUDA 구현은 시작하지 않았다**.
이 지시가 과거 CPU benchmark 재개 계획과 “CUDA 범위 밖” 메모보다 우선한다.

## 가장 먼저 읽을 문서

1. `AGENTS.md`
2. [CUDA forward 상세 계획](design/cuda-forward.md): CPU/GPU 경계, API/소유권, 메모리,
   연산별 구현, 정확도/오류/trace, 빌드, C0–C6 단계와 통과 조건.
3. [관측된 CUDA 환경](../manifests/cuda-environment.json)
4. [기존 수치/형식 계약](design/contracts.md)
5. 필요 시 [이전 CPU handoff](CPU_HANDOFF_2026-09-22.md)와 [CPU 결과](results.md)

## 다음 세션에서 할 일

- 실제 `git status --short --branch` / `git log -3 --oneline`부터 확인한다.
- 정상 작업 디렉터리는 `/home/rmj/FPGA_LLM/llama2.c`다.
- 로컬 변경을 보호하고 CUDA 구현 stage branch에서 시작한다.
- **C0: 작은 C→CUDA wrapper→kernel/cuBLAS 실행 확인**부터 수행한다.
- **C1: C ABI, Model/Session 소유권, 선택적 CUDA build 기반**을 만든다.
- 첫 milestone은 FP32 weights/FP32 KV의 전체 GPU forward와 기존 C CLI 연결이다.
- tokenizer, sampling, FP64 scoring, 파일 검증은 CPU에 유지한다.
- weights/KV/scratch는 GPU에 상주시킨다. layer마다 CPU로 activations를 가져오지 않는다.
- CPU `SmKernelOps`는 host pointer 전용이며 CUDA backend ABI로 재해석하지 않는다.
- 기존 CPU `SmSession`을 유지하고 별도 CUDA opaque Model/Session + 상위 dispatch를 기본안으로 한다.
- W8A8/KV8, TF32/BF16, FlashAttention, 성능 최적화는 FP32 완료 뒤 별도 단계다.

## 이번 세션에서 실제 확인한 것

- GPU: RTX4070Ti, compute capability8.9, driver591.86, 총 VRAM12,282MiB.
- `/usr/bin/nvcc`: CUDA11.5 / V11.5.119. `sm_89` target을 지원 목록에서 찾을 수 없다.
- Runtime/cuBLAS11 계열 library가 loader 목록에 있다. 실제 실행/로드 경로 검증은 아직이다.
- 기존 toolkit에서 `sm_86` cubin + `compute_86` PTX 경로를 먼저 검증하는 안을 기록했다.
  `-arch=sm_89`를 현재 nvcc에 바로 넣지 않는다. 전역 toolkit/driver를 임의로 교체하지 않는다.
- 실행한 것은 source/문서 읽기와 nvidia-smi/nvcc/ldconfig metadata 조회뿐이다.
- **새 모델 실행, benchmark, full 평가, CUDA build/kernel 실행은 없었다.**
- 이 세션에서 시작해 계속 돌아가는 장시간 process나 subagent는 없다.

## 유지해야 할 기준

- SmolLM2-135M Base revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` 고정.
- 모델/토크나이저/자료/HF 참조는 기존 local artifacts를 재사용한다.
- 기존 CPU 검증: 143 tests 통과, 수치 matrix 완료, 제한된 성능4조건/smoke12조건 완료.
  이 세션에서 위 검증을 다시 실행한 것은 아니다.
- CPU binary SHA256:
  `32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`.
- CPU 코드 기준 commit은 `26ff391`, 엔진 소스는 `134321d`와 같다.
- FP32 logit `atol=1e-3, rtol=1e-4`, 고정 probe 평균 NLL 차이 `<=1e-4`를 유지한다.
  CUDA gate 실패는 분석하며 미리 tolerance를 늘리지 않는다.
- 사용자 요청은 성능 측정 중단이다. CUDA correctness용 작은 tests/probe 검증은 필요하지만
  이전 full 평가나 24조건 성능 matrix를 자동 재개하지 않는다.
- 기존 30분 제한이 측정을 다시 하라는 뜻은 아니다. 이번 방향에서는 새 benchmark가 필요 없다.
- 기존 결과와 `manifests/measurement-plan.json`은 CPU 단계의 역사적 증거로 유지한다.
- Skill/agent는 프로젝트 로컬만 사용한다. 전역 설정과 프로젝트 `.codex`는 수정하지 않는다.
- CPU 전용 skill의 CUDA 제외 문구는 이전 단계 범위다. 최신 사용자 CUDA 요청이 우선한다.
- 위임은 실제 세션의 위임 규칙을 따르며 계획서의 파일 표만 보고 agent를 자동 생성하지 않는다.

## 재개 요청 예시

> `/home/rmj/FPGA_LLM/llama2.c/docs/HANDOFF.md`와 `docs/design/cuda-forward.md`를 읽고
> CUDA FP32 forward의 C0/C1부터 구현해줘. 추가 benchmark/full 평가는 필요 없어.
> 기존 CPU 동작과 수치 계약을 유지하고 Skill/agent 설정은 프로젝트 로컬에서만 관리해.

Git 통합/push는 기존 승인 범위 안에서 검증된 단계별로 진행한다. force push는 하지 않는다.
현재 계획 문서의 실제 반영 commit/branch는 `git log`와 `git status`로 확인한다.
