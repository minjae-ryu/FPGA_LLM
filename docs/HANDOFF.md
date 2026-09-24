# CPU 기준 복원 — 다음 세션 handoff

2026-09-24 최신 사용자 지시: **당분간 CUDA를 고려하지 않는다. CUDA 이전 Git
상태로 돌아가고 handoff 문서를 남긴다.** 이 지시가 CUDA 구현·속도 비교 계획보다
우선한다. CUDA 구현, GPU 측정, 미완료 비교를 자동으로 재개하지 않는다.

## 현재 작업 기준

- 저장소: `/home/rmj/FPGA_LLM/llama2.c`.
- 복원 기준: **`26ff391`**, CUDA 계획 이전의 마지막 CPU 검증/결과 commit.
- CPU 엔진 소스 기준: `134321d78ecec1ccdee0cdd3377b677030af7905`.
  `26ff391`은 같은 엔진 소스에 최종 CPU 검증과 결과를 기록한 commit이다.
- main은 이력을 지우지 않고 CUDA C0/C1(`2201faf`)과 계획(`2655aeb`)을
  되돌리는 새 commit으로 복원했다. 따라서 HEAD 자체는 `26ff391`이 아니지만,
  **코드·공개 header·Makefile·scripts·tests는 해당 CPU 기준과 동일**하다.
  현재 범위와 복원 기록을 설명하는 문서/manifest만 새로 갱신했다.
- 실제 HEAD/원격 반영 상태는 `git status --short --branch`, `git log -4 --oneline`으로 확인한다.

## 이번 복원에서 실제 검증한 것

- 새 `build/`에서 `make smollm check`: **143 passed in 8.46s**.
- `manifests/engine-build.json`에 기록된 source/build 파일 **20개 SHA256 일치**.
- 재빌드한 `build/smollm`도 기존 CPU binary와 SHA256이 일치한다:
  `32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`.
- 이번 복원에서 새로운 모델 benchmark, full 평가, GPU 실행은 하지 않았다.
  기존 numerical matrix·sanitizer·성능·smoke 결과는 과거 CPU 단계의 증거다.
- CUDA 비교 프로세스가 남아 있지 않음을 샌드박스 밖에서 확인했다.

자세한 복원 절차: [Task 13](tasks/13-restore-cpu.md).
기계 판독 기록: [cpu-restore.json](../manifests/cpu-restore.json).

## 보존한 CUDA 작업 — 현재 작업 대상 아님

- 보존 브랜치: **`archive/cuda-paused-2026-09-24`**.
- 보존 commit: **`2d5ab02`**. CUDA full forward/CLI와 검증 결과 및 미완료 비교를
  포함한다. 중단 당시 benchmark는 **6조건 중5조건**만 저장돼 있었다.
  완성된 비교 결과로 해석하거나 나머지 조건을 자동 실행하지 않는다.
- 현재 CPU 작업 트리에서는 CUDA 소스/header/target/test/계획을 제거했다.
  필요할 때 Git 보존 브랜치에서 볼 수 있으며, 사용자 재요청 전에는 다시 가져오지 않는다.
- Git에서 제외된 CUDA build, JIT cache, probe trace는 로컬
  `artifacts/cuda-paused-2026-09-24/`로 옮겼다. 이 경로는 원격에 올라가지 않는다.
  현재 `build/`는 CPU 기준으로 새로 생성했다.
- 시스템 driver/toolkit과 전역 설정은 변경하지 않았다.

## 재개할 때 읽을 문서

1. `AGENTS.md`: 최신 CPU 전용 범위와 운영 규칙.
2. [contracts.md](design/contracts.md): 형식·수치·scoring 계약.
3. [CPU 결과](results.md), [최종 CPU 검증 기록](tasks/09-final-matrices.md).
4. [이전 CPU handoff 원본](CPU_HANDOFF_2026-09-22.md): 구현·산출물 상세.
   과거의 “남은 full 평가/성능 matrix”는 역사적 계획이며 현재 실행 지시가 아니다.
5. [validation.md](validation.md): 필요할 때 사용할 검증 명령.

## 현재 CPU 구현과 완료 결과

- SmolLM2-135M Base revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` 고정.
- C Model/Session, GQA9/3, split-half RoPE, chunk prefill GEMM/decode GEMV,
  NONE/LAST/ALL, KernelOps/MathOps, W8A8 GS64, post-RoPE KV8,
  tokenizer, sampling, FP64 scoring, trace/replay, C 결과 비교 구현.
- CLI: `inspect`, `generate`, `logits`, `chunks`, `eval`, `bench`, `replay`, `compare-results`.
- 기존 CPU numerical matrix: logits8, chunk/reset32, HF 중간값21, math19,
  linear88, 누적 trace222개 aggregate. FP32 gate 통과, math19개 bit-exact.
- 기존 제한 성능4조건, smoke평가12조건, smoke비교9행 완료.
  **전체 split 평가와 정식24조건 성능 matrix는 미완료**이며 지금 재개하지 않는다.

## 재사용할 로컬 파일

- `models/smollm2-f32.bin`, `models/smollm2-q8.bin`, `models/tokenizer.bin`.
- `artifacts/reference/`: 기존57/315토큰 HF logits 및 중간 tensor.
- `data/`: WikiText/HellaSwag/PIQA의 smoke/full 준비 파일.
- `traces/numerical/`, `results/scores/`: 기존 CPU trace와 세부 scoring 결과.
- `results/benchmarks-small.*`, `results/evaluation-smoke*`, `results/numerical-*`:
  기존 CPU 집계 결과. 원래 measurement manifest/checkpoint도 보존했다.

기존 artifact를 재사용한다. 모델·자료를 다시 다운로드할 필요가 없다.
새 코드가 필요하면 source/binary/input hash를 구분하며 과거 결과와 이어 붙이지 않는다.
FP32 logit `atol=1e-3, rtol=1e-4`, 평균 NLL 차이 `<=1e-4`를 유지한다.
Skill/agent 설정은 프로젝트 로컬만 사용한다. 전역 설정이나 `.codex`를 수정하지 않는다.

## 다음 작업

CPU 기준 복원과 handoff는 완료했다. **새 CPU/FPGA 구현 목표는 아직 지정되지 않았다.**
다음 사용자 요청에 맞춰 진행하며 CUDA, 추가 benchmark/full 평가를 임의로 시작하지 않는다.

재개 요청 예시:

> `/home/rmj/FPGA_LLM/llama2.c/docs/HANDOFF.md`를 읽고 CPU 기준에서 작업을 이어가자.
> CUDA는 고려하지 말고, 다음으로 [구체적인 CPU/FPGA 작업]을 해줘.
