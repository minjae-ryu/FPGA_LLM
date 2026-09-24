# 보존 브랜치 — CUDA 작업 중단

2026-09-24 사용자가 CUDA를 당분간 고려하지 않고 CUDA 이전 CPU 코드로
돌아가도록 요청했다. 이 브랜치는 미완료 CUDA forward/비교 작업의 보존본이다.
CPU 작업 재개는 main의 docs/HANDOFF.md를 따른다.

- CPU 복원 기준: 26ff391.
- CUDA C0/C1: 2201faf.
- 이 보존본: GPU full forward 및 CLI; synthetic/실제 HF gate 통과.
- CUDA 비교: 6조건 중5조건 저장, 미완료. 자동 재개 금지.
- 마지막 확인에서 CUDA 비교 프로세스 없음.
- 상세: docs/tasks/12-cuda-forward-comparison.md, manifests/cuda-forward-comparison.json.
- Ignore된 build/cache/trace/model은 Git 보존 대상이 아니다.
