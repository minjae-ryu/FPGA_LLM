# SmolLM2 C 실험 — 세션 handoff

2026-09-22 재개 작업에서 최종 빌드 검증과 제한된 첫 비교를 완료했다.
사용자가 도중에 **벤치마크 개수를 줄이고 총 측정 시간을 최대 30분으로 제한**했다.
이 최신 범위가 이전의 전체 matrix 실행 계획보다 우선한다.
**전체 split 평가와 정식 24조건 성능 matrix는 완료하지 않았다.**

## 현재 완료 범위

- `make smollm check`: **143 passed in 8.72s**. 마지막 JSON parser 회귀까지 포함한다.
- 최종 실행 파일 SHA256:
  `32c208d95e7d399d60c7f87981833331dc0514eef05451e33a68609028d1f85e`.
- 이 파일로 numerical matrix를 실제 재실행했다. logits 8, chunk/reset 32,
  HF 중간값 21, math 19, linear 88, 누적 trace 222개 aggregate.
  FP32 gate와 HF 중간값 gate 통과, math 19개 모두 bit-exact.
- 원래 수치 결과의 값은 다시 측정한 값과 동일하다. 바이너리 hash만 고친 것이 아니다.
  `manifests/engine-build.json`에 source별 hash, compiler/CPU/OpenBLAS/PCRE2와 증거 기록.
- **소규모 성능 4조건** 완료: 네 구성 × prompt128 × thread1,
  decode128, warmup2, 측정5회 median. 모두 fresh process, 다른 모델 실행 없이 직렬 측정.
- **Smoke 평가 12조건** 완료: 네 구성 × WikiText/HellaSwag/PIQA.
  WikiText 8,191 targets/13 windows, HellaSwag/PIQA 각각 최초32문제.
  jobs4, threads3, chunk128, context2048. 병렬 평가 시간은 성능에 사용하지 않는다.
- threads3에서 FP32 57/315토큰 reference gate를 먼저 확인했다.
  최대 logit 오차 `9.44137573242e-05` / `0.000351905822754`,
  평균 NLL 차이 `-3.04112805034e-07` / `6.89774502088e-08`, top-1 변화 0.
- C `compare-results`의 smoke 비교 9행과 보고서 생성 완료.
- 원래 성능 matrix는 11조건 완료 후 사용자 범위 변경에 따라 SIGINT로 중단했다.
  `results/benchmarks-extended-partial.jsonl/csv`에 보존했다.
  소규모 matrix는 그중 동일한2조건을 재사용하고 KV8의2조건만 추가 측정했다.

측정 예산은 원래 성능 시작보다 이른 **21:46 KST**부터 보수적으로 합산했다.
공통 deadline은 **22:16 KST**였고, 실제 마지막 모델 실행은 **22:04:33.121 KST**에
끝났다. 총 **1,113.121초(약18분33초)**로 30분 이내다. 후속 명령에는 남은 예산을
계산한 GNU `timeout`을 적용했으며 timeout이 발동하기 전에 모두 완료됐다.
현재 이 세션이 시작한 모델/평가/성능 프로세스는 남아 있지 않다.

- 결과 보고서: [results.md](results.md)
- 상세 실행 기록: [tasks/09-final-matrices.md](tasks/09-final-matrices.md)
- 예산/범위/종료 시간: [measurement-plan.json](../manifests/measurement-plan.json)
- 요약 JSONL/CSV: `results/benchmarks-small.*`, `results/evaluation-smoke.*`,
  `results/evaluation-smoke-deltas.*`, `results/numerical-*`.
- 문서 재생성: `python3 scripts/render_results.py --scope small`.
  실제 C 결과만 서식화하며, delta는 C로 계산한다.

WikiText smoke PPL: FP32/FP32 **15.566465**, W8A8/FP32 **15.636703**,
FP32/KV8 **15.578888**, W8A8/KV8 **15.666988**.
32문제 HellaSwag와 PIQA에서는 네 구성의 정확도가 같았지만 전체 split 결과를
의미하지 않는다. 상세 오차/속도/메모리는 보고서를 확인한다.

## 다음 작업과 재개 주의

1. `git status --short --branch`와 `git log -3 --oneline`으로 실제 HEAD와 변경을 확인한다.
   정상 작업 저장소는 `/home/rmj/FPGA_LLM/llama2.c`다.
2. 최신 사용자 범위는 작은 비교와 최대30분이다. 이미 끝난 측정을 반복하거나
   전체 split 평가를 무제한으로 시작하지 않는다. 범위를 확장할 때도 시간 제약을 반영한다.
3. 남은 원래 범위는 네 구성의 **전체 WikiText test / HellaSwag validation / PIQA validation**,
   그리고 **4구성 × prompt128/512/2048 × thread1/12 = 24조건 성능 matrix**다.
   소규모 결과를 full 결과라고 표시하지 않는다.
4. 원래 성능 checkpoint는 `results/.benchmarks.checkpoint.json`에 11조건을 보존한다.
   `python3 scripts/run_benchmarks.py bench`는 hash/options가 같으면 여기서 재개하지만,
   먼저 허용된 측정 시간과 범위를 정한다. 평가 full은 아직 시작하지 않았다.
5. 현재 smoke 재개 명령은 아래와 같다. 모두 완료돼 있으므로 같은 옵션이면
   점검만 하고 모델을 다시 실행하지 않는다.

   ```sh
   python3 scripts/run_benchmarks.py eval --variant smoke --jobs 4 --threads 3 --progress-every 10
   ```

6. 모델 core/source/build를 바꾸면 FP32 gate부터 다시 확인한다. binary/model/data/
   manifest/options/hash가 다른 checkpoint에 결과를 이어 붙이지 않는다.
   새 output stem 또는 명시적 `--restart`를 사용한다.
7. 원래 전체 matrix가 나중에 모두 완성되면
   `python3 scripts/render_results.py --scope full`로 C 비교와 전체 보고서를 만든다.

## 사용자 의도와 운영 제약

- SmolLM2-135M **Base**, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` 고정.
- BF16 배포 가중치의 정확한 FP32 변환이 기준. 비교 구성은 FP32/FP32,
  W8A8 GS64/FP32, FP32/KV8, W8A8 GS64/KV8.
- 추론·평가·수치/결과 비교는 C, 변환·자료 준비·HF reference 생성은 Python.
- OpenBLAS/PCRE2 사용. CUDA, 직접 작성한 AVX, Taylor, 고정소수점 scale은 범위 밖.
  `-march=native`의 컴파일러 자동 벡터화는 사용한다.
- **Skill과 subagent 설정은 프로젝트 로컬에서만 관리한다.**
  `.agents/skills/`, `AGENTS.md`, `docs/agents/`, `docs/tasks/`를 사용한다.
  전역 skill/agent 설정과 `.codex`를 변경하지 않는다. 이번 재개에서 전역 변경은 없었다.
- 메인 `gpt-6-astra`/`max`, worker·reviewer `gpt-5.6-sol`/`xhigh`, 최대 메인1+worker3.
  실제 위임 시 프로젝트/세션의 위임 지침을 따르고 모델을 명시한다.
  이번 재개 작업은 subagent 없이 수행했다.
- 공통 API/형식/빌드/Git은 메인 소유. worker에게는 겹치지 않는 파일 범위를 배정한다.
- FP32 logits `atol=1e-3`, `rtol=1e-4`; probe 평균 NLL 차이 `<=1e-4`.
  실패를 숨기기 위한 tolerance 완화 금지. Q8 정확도에는 임의 합격선이 없다.
- 성능 측정은 다른 모델 실행·무거운 테스트와 겹치지 않는다.

계약: [design/contracts.md](design/contracts.md).
실행 명령: [validation.md](validation.md).

## 구현과 기존 검증

C Model/Session, GQA9/3, head64, split-half RoPE, chunk prefill GEMM/decode GEMV,
NONE/LAST/ALL projection, KernelOps/MathOps, GS64 int8+FP32 scale, post-RoPE KV8,
PCRE2 tokenizer, sampling, FP64 scoring, replay, C 결과 비교까지 구현됐다.
전체 FP32 KV shadow는 없다. mmap loader는 malformed tensor/config/Q8 block을 거부한다.
CLI는 `inspect`, `generate`, `logits`, `chunks`, `eval`, `bench`, `replay`,
`compare-results`를 제공한다. 런타임 소스는 checkpoint
`134321d78ecec1ccdee0cdd3377b677030af7905`와 동일하다.

기존 세션의 다음 증거는 변경 없는 소스에 대해 유지된다:

- legacy `make run testcc`: ALL OK; `pytest -q test_all.py`: 2 passed.
- cache/kernel ASan+UBSan+LSan 통과. LSan은 sandbox ptrace 제약 때문에 승인된
  외부 실행에서 검증했다. replay/tokenizer sanitizer도 통과.
- BF16→FP32 134,515,008개 값 bit 비교: 불일치0.
- tokenizer seeded fuzz2,000개 ID 일치, 전체49,152 ID decode 검증.
- 실제 모델 chunk1/127/128/129, 혼합 분할, reset, synthetic causal/GQA/context 검증.
- Q8 chunk 차이에 FP32 end-to-end tolerance를 적용하지 않는다. 작은 GEMM/GEMV
  차이가 quantizer 경계를 넘는 원인은 [task05](tasks/05-independent-review.md)에 기록돼 있다.
  fixed-input 양자화 정수와 cache는 여전히 정확 일치가 필요하다.

## 재사용 가능한 산출물 — Git 제외

- `models/smollm2-f32.bin`: 538,095,104 bytes,
  SHA `2d2367f33b1e87706b959fa8ec1de4b3d04572c5d5bad9b0fd6ecbd905fc7849`.
- `models/smollm2-q8.bin`: 143,060,480 bytes,
  SHA `b9ab3f3690b6648fe3ba2c1b75e1cb9af93b54d6f4ce9ad93867d20c2e61873d`.
- `models/tokenizer.bin`: 1,683,200 bytes,
  SHA `c2145cd851d7d1614d206f825f805262c97c0b81c63f987ab5ec2552da066ad2`.
- `artifacts/reference/`: HF probe/long token/logit 및 중간 tensor.
- `data/{wikitext2,hellaswag,piqa}_{smoke,full}.smeval`: 6개 모두 준비돼 있다.
- `traces/numerical/`: 최종 바이너리의 네 구성 trace.
- `artifacts/export/`: hd64/dim192/GQA3/1/context384 synthetic 모델.
- `artifacts/review/chunk_sensitivity.json`: 독립 Q8 원인 진단.
- `results/scores/`: 완료된 smoke 평가 per-record 결과. checkpoint와 함께 Git 제외.
- `test/`: 기존 tinyllama 공식260K 모델과 tok512 회귀 자료.

소스 HF snapshot은 기존 캐시
`/home/rmj/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
에서 읽기만 했다. 재다운로드할 필요가 없다. 데이터 revision, hash, harness commit과
정확한 Unicode choice normalization은 `manifests/` 및 [design/data.md](design/data.md) 참조.

## Git와 환경

- 상위 `/home/rmj/FPGA_LLM` 자체는 Git 저장소가 아니다.
- `origin`: `https://github.com/minjae-ryu/FPGA_LLM.git`.
- `upstream`: `https://github.com/karpathy/llama2.c`.
- 원래 FPGA main `8051b2ab782a4908dea802dded821c2cc4d765f3` 이력을 보존했다.
- 원본 llama2.c 보존 브랜치 `preserve/llama2c-350e04f`, 원본 전체SHA
  `350e04fe35433e6d2941dce5a1f53308f87058eb`.
- `/home/rmj/FPGA_LLM/setup-worktree`는 초기설정 브랜치다. 정상 작업은 여기서 하지 않는다.
- 검증된 단계 커밋을 main에 정상 통합하고 push하는 것은 기존 사용자 승인 범위다.
  원격 fetch 후 확인하고 force push하지 않는다. 실제 반영 상태는 Git으로 확인한다.
- GitHub OAuth에 `workflow` scope가 없으므로 원본 Actions는
  `docs/llama2c-build-workflow.yml`로 보존했다. 새 workflow를 무심코 추가하지 않는다.
- CPU AMD Ryzen9 7900, 12 cores/24 logical CPUs, WSL2 Linux,
  OpenBLAS0.3.20, PCRE2 10.39, GCC11.4.0. CUDA를 사용하지 않았다.
- Python은 `/home/rmj/SRAM/venv/bin/python3`를 읽기/실행만 사용했다.
  전역 환경을 수정하지 않았다. 패키지 버전은 `requirements-smollm.txt` 참조.
