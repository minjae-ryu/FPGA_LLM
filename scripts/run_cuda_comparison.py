#!/usr/bin/env python3
"""Bounded serial FP32 CPU/CUDA comparison; C computes inference, scores and timings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["correctness","bench"],required=True)
    args=parser.parse_args()
    os.chdir(ROOT)
    out=ROOT/"results/cuda-fp32"
    out.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/"manifests/cuda-forward-comparison.json"
    binary=ROOT/"build/smollm-cuda"
    model=ROOT/"models/smollm2-f32.bin"
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [binary,model,
        ROOT/"artifacts/reference/probe.tokens",ROOT/"artifacts/reference/probe.logits",
        ROOT/"artifacts/reference/long.tokens",ROOT/"artifacts/reference/long.logits"]}
    env=dict(os.environ,CUDA_CACHE_PATH=str(ROOT/".cache/cuda"))
    env.pop("CUDA_FORCE_PTX_JIT",None);env.pop("CUDA_DISABLE_PTX_JIT",None)
    def run(command,backend,threads,extra):
        cmd=[str(binary),command,"--model",str(model),"--backend",backend,"--threads",str(threads),*extra]
        print(f"{command} {backend} threads={threads} {' '.join(extra)}",flush=True)
        proc=subprocess.run(cmd,env=dict(env,OPENBLAS_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads)),
            capture_output=True,text=True,timeout=600)
        if proc.returncode:
            raise RuntimeError(f"gate/run failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}")
        return [dict(json.loads(line),argv=cmd) for line in proc.stdout.splitlines() if line]
    if args.phase=="correctness":
        # Verify the original pinned HF artifact identities before executing gates.
        for name in ("probe","long"):
            ref=json.loads((ROOT/f"manifests/reference_{name}.json").read_text())
            for kind in ("tokens","logits"):
                info=ref["artifacts"][kind]
                assert digest(ROOT/info["path"])==info["sha256"],info["path"]
        rows=[]
        (ROOT/"traces").mkdir(exist_ok=True)
        for backend,threads in [("cuda",1),("cpu",1),("cpu",12)]:
            for name in ("probe","long"):
                extra_trace=["--trace","traces/cuda-probe.smtrace"] if backend=="cuda" and name=="probe" else []
                result=run("logits",backend,threads,["--tokens",f"artifacts/reference/{name}.tokens",
                    "--reference",f"artifacts/reference/{name}.logits","--context","512","--chunk","128",*extra_trace])
                assert result[0]["parity_pass"] is True
                rows+=result
        replay=[str(binary),"replay","--mode","compare","--trace","traces/cuda-probe.smtrace","--reference-prefix","artifacts/reference/probe"]
        proc=subprocess.run(replay,env=env,capture_output=True,text=True,timeout=60)
        if proc.returncode: raise RuntimeError(proc.stderr+proc.stdout)
        intermediate=[dict(json.loads(line),argv=replay) for line in proc.stdout.splitlines()]
        assert len(intermediate)==21 and all(row["parity_pass"] for row in intermediate)
        rows+=intermediate
        rows+=run("chunks","cuda",1,["--tokens","artifacts/reference/long.tokens","--context","512"])
        (out/"correctness.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        state={"updated_at_utc":datetime.now(timezone.utc).isoformat(),"hashes":hashes,
            "correctness":"passed","correctness_sha256":digest(out/"correctness.jsonl"),
            "scope":"FP32 weights/KV; six benchmark cells; no full evaluation or quantized matrix",
            "benchmark_status":"not run","environment":{"LD_LIBRARY_PATH":env.get("LD_LIBRARY_PATH",""),
                "CUDA_CACHE_PATH":env["CUDA_CACHE_PATH"]}}
        manifest.write_text(json.dumps(state,indent=2)+"\n")
    else:
        state=json.loads(manifest.read_text())
        assert state["correctness"]=="passed" and state["hashes"]==hashes,"Correctness gates required for this exact build/inputs"
        assert digest(out/"correctness.jsonl")==state["correctness_sha256"],"Correctness evidence changed"
        # Fresh process per cell, strictly serial; no other tests/model runs in parallel.
        with (out/"benchmark.jsonl").open("w") as f:
            for prompt in (128,512):
                for backend,threads in [("cpu",1),("cpu",12),("cuda",1)]:
                    row=run("bench",backend,threads,["--prompt",str(prompt),"--decode","128","--warmup","2",
                        "--repetitions","5","--context","1024","--chunk","128"])[0]
                    assert row["backend"]==backend and row["logit_delivery"]=="host_last_row"
                    f.write(json.dumps(row)+"\n");f.flush()
        state.update(benchmark_status="complete",benchmark_sha256=digest(out/"benchmark.jsonl"),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),warmups=2,repetitions=5,
            cells=6,timing="synchronous host wall clock, LAST logits returned to host, median; excludes load/upload/session setup")
        manifest.write_text(json.dumps(state,indent=2)+"\n")

if __name__=="__main__":
    main()
