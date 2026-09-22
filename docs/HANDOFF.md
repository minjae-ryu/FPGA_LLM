# SmolLM2 C 실험 — 세션 handoff

2026-09-22 사용자가 현재 시점에서 handoff 후 새 세션 재개를 요청했다.
새 장시간 작업을 시작하지 않고 현재 코드를 작업 브랜치에 checkpoint한다.
**전체 프로젝트 완료 상태가 아니다.** 구현·핵심 수치 검증은 끝났고,
전체 split 평가와 정식 성능 matrix가 남아 있다.

## 새 세션의 첫 작업

```sh
cd /home/rmj/FPGA_LLM/llama2.c
git status --short --branch
cat AGENTS.md
cat docs/HANDOFF.md
make smollm check
```

먼저 handoff 이후 변경 여부를 확인한다. 원래 사용자 목표를 다시 설계하거나
기존 산출물을 재다운로드하지 말고 아래 남은 작업에서 이어간다.
로컬 checkpoint의 커밋은 `git log -1`로 확인한다.

## 사용자 의도와 운영 제약

- SmolLM2-135M **Base** 고정 revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`를 C에서 실행한다.
- BF16 배포 가중치의 정확한 FP32 변환을 기준으로 FP32/FP32,
  W8A8 GS64/FP32, FP32/KV8, W8A8 GS64/KV8 네 구성을 비교한다.
- 추론·평가·수치/결과 비교는 C, 변환·자료 준비·HF 기준 생성은 Python.
- OpenBLAS/PCRE2 사용. CUDA, 직접 작성한 AVX, Taylor, 고정소수점 scale은
  이번 범위 밖이다. `-march=native`에 의한 컴파일러 자동 벡터화는 사용한다.
- **Skill과 subagent 설정은 프로젝트 로컬에만 둔다.** 사용자가 이를 다시
  강조했다. `.agents/skills/`, `AGENTS.md`, `docs/agents/`, `docs/tasks/` 사용.
  전역 skill/agent 설정과 `.codex`를 변경하지 않는다.
- 메인 `gpt-6-astra`/`max`, worker·reviewer `gpt-5.6-sol`/`xhigh`, 최대 메인1+worker3.
  모델 지정은 새 worker 생성 요청에 명시한다. 기존 세션 agent ID에 의존하지 않는다.
- 공통 API/형식/빌드/Git은 메인 소유, worker 파일 범위는 겹치지 않게 배정한다.
- FP32 logits `atol=1e-3`, `rtol=1e-4`; 고정 probe 평균 NLL 차이 `<=1e-4`.
  실패를 숨기기 위해 허용 오차를 완화하지 않는다. Q8 정확도에는 임의 합격선이 없다.
- 성능 측정은 다른 모델 실행·무거운 테스트와 **동시에 하지 않는다**.

수치/API/파일 계약: [design/contracts.md](design/contracts.md).
실행 명령: [validation.md](validation.md).

## Git 상태

- 실제 작업 저장소: `/home/rmj/FPGA_LLM/llama2.c`.
  상위 `/home/rmj/FPGA_LLM` 자체는 Git 저장소가 아니다.
- 현재 브랜치: `stage/fp32-baseline`.
- `origin`: `https://github.com/minjae-ryu/FPGA_LLM.git`.
- `upstream`: `https://github.com/karpathy/llama2.c`.
- 원래 FPGA main `8051b2ab782a4908dea802dded821c2cc4d765f3`의 이력을 유지했다.
  별도 worktree에서 추적된 Verilog를 제거하고 llama2.c와 로컬 설정으로 교체했다.
- 원격 main에 반영된 것은 초기 교체 두 커밋까지다:
  `fe019dd` (교체/설정), `4e9f6c5` (원본 workflow 문서 보존).
- **이번 엔진 구현/checkpoint는 아직 main에 통합하거나 원격에 push하지 않았다.**
- GitHub OAuth 인증에 `workflow` scope가 없어 원본 Actions 파일 push가 거절되어,
  `docs/llama2c-build-workflow.yml`로 옮겼다. 새 workflow를 무심코 추가하지 않는다.
- 원본 llama2.c 보존 브랜치 `preserve/llama2c-350e04f`, 원본 전체 SHA
  `350e04fe35433e6d2941dce5a1f53308f87058eb`.
- 초기 작업용 worktree `/home/rmj/FPGA_LLM/setup-worktree`는
  `setup/smollm-foundation` / `fe019dd`에 남아 있다. 정상 작업은 여기서 하지 않는다.
- force push 없이 검증된 단계 커밋을 main에 통합하고 push하는 것까지 사용자 승인 범위다.

## 구현된 내용

- `include/smollm.h`: Model/Session, NONE/LAST/ALL callback, CPU KernelOps,
  call-site별 MathOps, trace, cache 메모리 계측, FP64 scoring, sampling.
- `src/model.c`: mmap 로더, 명시적 little-endian 형식/설정/tensor 검증,
  shape/name/dtype/offset/겹침/비유한수/비정상 Q8 block 거부.
- `src/session.c`: GQA9/3, head64, split-half RoPE, RMSNorm, SiLU,
  chunk prefill GEMM, decode GEMV, causal attention, position/reset/bounds.
- `src/kernels.c`, `cache.c`, `math_ops.c`, `scoring.c`, `sampling.c`.
  Q8 그룹은 int8 64개+FP32 scale, ties-to-even, [-127,127].
  KV8은 post-RoPE K와 V를 저장하고 현재 chunk도 복원 규칙을 거친다.
  전체 FP32 KV shadow는 없다.
- C 토크나이저: `src/tokenizer.c`, PCRE2, Digits→ByteLevel→BPE,
  special token, 자동 BOS/EOS 없음, 빈 생성만 token0.
- C CLI: `inspect`, `generate`, `logits`, `chunks`, `eval`, `bench`, `replay`,
  `compare-results`. 잘못된/중복/누락 옵션을 거부한다.
- `replay`: HF 중간값 비교, 동일 입력 math replay, 선형층
  FP32/weight-only/activation-only/W8A8 오차, 구성 간 누적 trace 비교.
- `compare-results`: 평가 JSONL을 C에서 파싱하고 동일한 데이터/채점 수/옵션을
  확인한 뒤 NLL/PPL/정확도 delta를 C로 계산한다.
- `scripts/export_smollm.py`, `export_tokenizer.py`, `prepare_reference.py`,
  `prepare_data.py`: 고정 자료와 manifest 준비.
- `scripts/run_numerical.py`: 계산은 C에 맡기고 결과에 hash/실행 정보를 붙인다.
- `scripts/run_benchmarks.py`: fresh-process 평가/성능 실행, JSONL/CSV,
  엄격한 hash 기반 resume. 평가만 `--jobs` 병렬 실행을 지원한다.
  성능은 항상 직렬이며 `--jobs`를 받지 않는다.
- legacy `run.c`, `runq.c`, `test.c`, `test_all.py`, LICENSE 원문 일치 확인 완료.

## 검증 상태 — 정확히 구분할 것

- 마지막 전체 `make smollm check`: **141 passed in 8.20s**.
- 그 뒤 결과 JSON parser에 escaped U+0000와 잘못된 raw UTF-8 거부 검사가
  추가됐다. 해당 focused suite **29 passed**, integrated build와
  `-fanalyzer`는 통과했다. **이 두 회귀 추가 후 전체 suite는 아직 다시 돌리지 않았다.**
  새 세션에서 첫 `make smollm check`를 수행한다.
- `make run testcc`: `ALL OK`; legacy `pytest -q test_all.py`: **2 passed**.
- cache/kernel harness ASan+UBSan+LSan 통과. LSan은 샌드박스의 ptrace 제한으로
  실패했으나 승인된 외부 실행에서는 통과했다. 별도 replay/tokenizer sanitizer도 통과.
- tokenizer: 2,000 seeded fuzz 입력 ID 정확 일치, 전체 49,152 ID decode 검증.
- BF16→FP32: 134,515,008개 값 bit 비교, 불일치 0.
- 실제 HF FP32/eager 비교:

| probe | tokens | 최대 logit 절대오차 | 평균 NLL 차이 | top-1 변화 |
|---|---:|---:|---:|---:|
| 짧은 자연어 | 57 | 8.34465e-5 | 5.76791e-7 | 0 |
| 긴 자연어 | 315 | 2.74658e-4 | 2.34769e-7 | 0 |

- thread12 긴 probe도 통과: 최대 2.00272e-4, NLL 차이 6.04505e-8.
- 실제 모델 chunk1/127/128/129·혼합 분할·reset 통과.
  synthetic HF 참조는 causal/GQA/context 한계와 오류 후 position 불변도 검증한다.
- 독립 C int32 dot/scale/양자화/cache 검증 통과.
- **Q8 chunk 차이에 FP32 end-to-end tolerance를 적용하지 않는다.**
  sub-1e-5 GEMM/GEMV 차이가 뒤 quantizer 경계를 넘어 커지는 것을 독립 추적했다.
  fixed-input 양자화 정수/cache는 여전히 정확 일치가 필요하다.
  [tasks/05-independent-review.md](tasks/05-independent-review.md)에 수치/위치/원인 기록.
- 검토에서 찾은 all-zero int8 + nonzero scale, negative-zero scale 로딩 문제는 수정됐다.

### 실행 완료된 수치 matrix

`python3 scripts/run_numerical.py` 완료, `results/numerical-*` JSONL/CSV에 보존.

- logits 8개 (4구성×57/315토큰)
- chunk/reset 집계 32개
- HF 중간 tensor 집계 21개 모두 통과
- math 집계 19개 모두 bit-exact
- 선형 22개 tensor × 4variant = 88개 집계
- 누적 trace 비교 222개 집계

57토큰 probe의 구성별 결과:

| 구성 | 평균 NLL | 평균 KL(HF 기준) | top-1 변화 |
|---|---:|---:|---:|
| FP32/FP32 | 4.0909824369 | 1.81333e-11 | 0 |
| W8A8/FP32 | 4.1961196810 | 0.04644756 | 4 |
| FP32/KV8 | 4.0885334211 | 0.00050102 | 1 |
| W8A8/KV8 | 4.1892754076 | 0.04700081 | 9 |

Q8 수치는 진단 결과이며 합격/불합격 문턱이 아니다.

**Build provenance 주의:** 수치 matrix와 `manifests/engine-build.json`의 실행 파일
SHA는 `f05906efefe2cee8cd8652a83b7c795128abef00ec21aec6ad7dfa318a7304e9`다.
이후 CLI 옵션 검증과 결과 비교 명령이 추가되어 현재 실행 파일 SHA는 다르다.
현재 SHA는 `manifests/handoff.json`에 기록했다. 모델 수치 core는 바뀌지 않았지만,
최종 배포 build의 일관된 증거가 필요하면 최종 build를 고정한 뒤 numerical matrix를
한 번 다시 실행하고 build manifest를 갱신한다. 이전 결과를 새 SHA로 단순 수정하지 않는다.

## 준비된 큰 산출물 — 재사용 가능, Git 제외

- `models/smollm2-f32.bin`: 538,095,104 bytes,
  SHA `2d2367f33b1e87706b959fa8ec1de4b3d04572c5d5bad9b0fd6ecbd905fc7849`.
- `models/smollm2-q8.bin`: 143,060,480 bytes,
  SHA `b9ab3f3690b6648fe3ba2c1b75e1cb9af93b54d6f4ce9ad93867d20c2e61873d`.
- `models/tokenizer.bin`: 1,683,200 bytes,
  SHA `c2145cd851d7d1614d206f825f805262c97c0b81c63f987ab5ec2552da066ad2`.
- `artifacts/reference/`: HF probe/long tokens, logits, 21 중간 tensor.
- `data/{wikitext2,hellaswag,piqa}_{smoke,full}.smeval`: 6개 모두 준비됨.
- `traces/numerical/`: 4구성 probe trace. `artifacts/replay/`에 초기 replay 자료.
- `artifacts/export/`: hd64·dim192·GQA3/1·context384 synthetic 두 형식.
- `artifacts/review/chunk_sensitivity.json`: 독립 Q8 chunk 원인 진단.
- `test/`: 기존 tinyllama 회귀용 공식 260K 모델과 tok512 자료.

모든 작은 재현 manifest는 `manifests/`에 있다. 소스 HF snapshot은 기존 캐시
`/home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
에서 읽기만 했다. 전체 source/config hash가 exporter에서 검증된다.

데이터 주의: HellaSwag/PIQA `acc_norm`은 **원래 처리된 choice의 Unicode 문자 수**다.
추가 delimiter나 context에서 옮긴 공백을 포함하지 않도록 수정/재생성했다.
WikiText test 304,986 tokens / 304,985 targets / 593 windows.
smoke는 8,192 tokens / 8,191 targets / 13 windows.
세부 revision과 harness commit은 [design/data.md](design/data.md) 및 각 manifest 참조.

## 남은 작업과 순서

1. 최종 전체 suite를 실행하고 source/binary를 고정한다.
   필요하면 현재 numerical matrix를 최종 build로 갱신하고
   `manifests/engine-build.json`에 실제 build/검증/CPU 정보 기록.
2. **정식 performance matrix는 아직 전혀 실행하지 않았다.**
   다른 heavy run이 없을 때 아래 명령 실행. 4구성×3prompt×2thread = 24 fresh processes.

   ```sh
   python3 scripts/run_benchmarks.py bench
   ```

   prompt128/512/2048, decode128, thread1/12, warmup2, measured5 median.
   성능 명령이 오래 걸릴 수 있다. 각 cell 완료 시 checkpoint가 갱신된다.
3. **네 구성의 완전한 smoke/full 평가 matrix는 아직 실행하지 않았다.**
   완료된 것은 FP32/FP32 WikiText smoke 한 번뿐이며
   `results/wikitext2-fp32-smoke.json`에 NLL2.7451189071/PPL15.5664647849,
   정확히8191 targets로 기록되어 있다. 이것을 전체 결과로 표현하지 않는다.

   ```sh
   python3 scripts/run_benchmarks.py eval --variant smoke
   python3 scripts/run_benchmarks.py eval --variant full
   ```

   기본 thread1, chunk128, context2048, 순차 실행. 필요하면 **정확도 평가만**
   `--jobs 3` 또는 `--jobs 4`, `--threads N`을 사용할 수 있다. 같은 matrix의
   thread 수는 맞추고 변경 시 FP32 기준 gate 확인/기록. 병렬 평가 시간은
   `concurrent_not_isolated`로 표시되고 성능 비교에 사용하지 않는다.
   단일 thread WikiText smoke가 약100초였으므로 full matrix는 오래 걸릴 수 있다.
   `--configurations`, `--tasks`, `--output-stem`으로 나누어 실행할 수도 있다.
4. 최종 aggregate를 C로 비교한다.

   ```sh
   build/smollm compare-results --input results/evaluation-smoke.jsonl > results/evaluation-smoke-deltas.jsonl
   build/smollm compare-results --input results/evaluation-full.jsonl > results/evaluation-full-deltas.jsonl
   ```

   Python은 CSV/문서 formatting만 담당한다. C가 NLL/PPL/acc delta를 계산한다.
5. 결과 요약 문서/README를 실제 전체 결과에 맞추어 작성하고,
   Q8 오차 기여, 누적 위치, PPL/정확도 변화, prefill/decode/TTFT/loading/RSS/KV/scratch를
   구분해 보고한다. summary JSONL/CSV와 manifest만 관리한다.
6. 검증된 브랜치를 main에 정상 통합/push. 원격 변경을 먼저 fetch해 확인한다.
   현재 checkpoint가 main에 이미 반영됐다고 가정하지 않는다.

Runner의 resume는 binary/model/data/manifest/options/hash가 모두 같을 때만 가능하다.
binary를 다시 빌드한 뒤 오래된 checkpoint에 결과를 이어 붙이지 않는다.
새 output stem 또는 명시적 `--restart`를 사용한다.
`results/scores/`와 `results/.*.checkpoint.json`은 Git에서 제외한다.

## 환경/실행 상태

- CPU: AMD Ryzen9 7900, 24 logical CPUs; 이 환경에 OpenBLAS0.3.20, PCRE2 10.39 설치.
- 약15GiB RAM, 수백 GiB disk 여유. CUDA를 사용하지 않는다.
- `python3`는 현재 `/home/rmj/SRAM/venv/bin/python3`이고 읽기/실행만 사용했다.
  torch2.6.0+cpu, transformers5.15.1, tokenizers0.22.2, numpy1.26.4 등은
  `requirements-smollm.txt`에 기록되어 있다. 전역 환경을 수정하지 않았다.
- GitHub와 HF 네트워크는 sandbox 안에서 DNS 실패. 필요한 fetch/push/download는
  기존 승인된 명령을 쓰거나 권한 tool로 요청한다. 승인을 우회하지 않는다.
- handoff 시점에 이 세션이 시작한 model/test/benchmark 프로세스는 모두 종료됐다.
  worker들도 마지막 보고를 끝냈다. 다음 세션에서 이어 기다릴 process/agent는 없다.

각 작업별 구체 증거는 `docs/tasks/00-setup.md`부터 `08-result-comparison.md`까지 참조.
