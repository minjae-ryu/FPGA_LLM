from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


HARNESS = r'''
#define _POSIX_C_SOURCE 200809L
#include "sm_cli.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct SmModel { int unused; };
struct SmSession { size_t position,context; };
static struct SmModel model;
static struct SmSession session;
static SmConfig config={192,64,2,3,1,64,4,4096,10000.0f,1e-5f,"stub-revision"};

const char *sm_option(int argc,char **argv,const char *key,const char *fallback) {
    for(int i=2;i<argc;i++)if(!strcmp(argv[i],key))return i+1<argc?argv[i+1]:NULL;
    return fallback;
}
int sm_flag(int argc,char **argv,const char *key) {
    for(int i=2;i<argc;i++)if(!strcmp(argv[i],key))return 1;
    return 0;
}
int sm_size_option(int argc,char **argv,const char *key,size_t fallback,size_t *out) {
    const char *value=sm_option(argc,argv,key,NULL);if(!value){*out=fallback;return sm_flag(argc,argv,key)?-1:0;}
    char *end=NULL;errno=0;unsigned long long parsed=strtoull(value,&end,10);
    if(errno||!value[0]||*end||parsed>SIZE_MAX)return -1;
    *out=(size_t)parsed;return 0;
}
int sm_run_open(SmRun *run,int argc,char **argv) {
    memset(run,0,sizeof(*run));run->model_path=sm_option(argc,argv,"--model",NULL);
    if(!run->model_path||sm_size_option(argc,argv,"--context",4096,&run->context)||
       sm_size_option(argc,argv,"--chunk",128,&run->chunk)){
        snprintf(run->error.message,sizeof(run->error.message),"bad run options");return -1;
    }
    size_t threads=1;if(sm_size_option(argc,argv,"--threads",1,&threads)||!threads)return -1;
    run->threads=(int)threads;run->model=&model;run->session=&session;run->load_seconds=.125;
    const char *stub_vocab=getenv("SM_STUB_VOCAB");
    config.vocab=stub_vocab?(uint32_t)strtoul(stub_vocab,NULL,10):4;
    run->kv=!strcmp(sm_option(argc,argv,"--kv","f32"),"q8")?SM_KV_Q8:SM_KV_F32;
    session.position=0;session.context=run->context;return 0;
}
void sm_run_close(SmRun *run){(void)run;}
double sm_time(void){static double now=10.0;now+=.01;return now;}
long sm_peak_rss_kib(void){return 321;}
void sm_json_string(FILE *out,const char *value){fputc('"',out);for(;*value;value++){if(*value=='"'||*value=='\\')fputc('\\',out);fputc(*value,out);}fputc('"',out);}
void sm_run_metadata(FILE *out,const SmRun *run){(void)run;fprintf(out,"\"model_revision\":\"stub\",\"linear\":\"fp32\",\"kv\":\"fp32\",\"threads\":1,\"chunk\":128,\"context\":4096");}
const SmConfig *sm_model_config(const SmModel *unused){(void)unused;return &config;}
SmDType sm_model_dtype(const SmModel *unused){(void)unused;return SM_F32;}
size_t sm_model_bytes(const SmModel *unused){(void)unused;return 111;}
size_t sm_session_kv_bytes(const SmSession *unused){(void)unused;return 222;}
size_t sm_session_scratch_bytes(const SmSession *unused){(void)unused;return 333;}
void sm_session_reset(SmSession *value){value->position=0;}
size_t sm_session_position(const SmSession *value){return value->position;}
int sm_prefill(SmSession *value,const uint32_t *tokens,size_t count,SmLogits mode,
               SmLogitCallback callback,void *opaque,SmError *error) {
    if(getenv("SM_STUB_FAIL")){snprintf(error->message,sizeof(error->message),"injected execution failure");return -1;}
    if(count>value->context-value->position){snprintf(error->message,sizeof(error->message),"context overflow");return -1;}
    static const float logits[4]={0,1,2,3};
    if(mode==SM_LOGITS_ALL&&callback)for(size_t i=0;i<count;i++)if(callback(opaque,value->position+i,logits,4))return -1;
    if(mode==SM_LOGITS_LAST&&callback&&count)if(callback(opaque,value->position+count-1,logits,4))return -1;
    for(size_t i=0;i<count;i++)if(tokens[i]>=4){snprintf(error->message,sizeof(error->message),"bad token");return -1;}
    value->position+=count;return 0;
}
int sm_decode(SmSession *value,uint32_t token,SmLogits mode,SmLogitCallback callback,void *opaque,SmError *error){return sm_prefill(value,&token,1,mode,callback,opaque,error);}

int main(int argc,char **argv){
    if(argc<2)return 2;
    if(!strcmp(argv[1],"eval"))return sm_command_eval(argc,argv);
    if(!strcmp(argv[1],"bench"))return sm_command_bench(argc,argv);
    return 2;
}
'''


def record(
    *,
    group: int = 0,
    label: int = 0,
    choice: int = 0,
    nchoices: int = 1,
    tokens: tuple[int, ...],
    score_start: int,
    score_end: int,
    norm_chars: int = 0,
) -> bytes:
    return struct.pack(
        "<8I",
        group,
        label,
        choice,
        nchoices,
        len(tokens),
        score_start,
        score_end,
        norm_chars,
    ) + struct.pack(f"<{len(tokens)}I", *tokens)


def smeval(kind: int, records: list[bytes]) -> bytes:
    return struct.pack("<8sII", b"SMEVAL01", kind, len(records)) + b"".join(records)


@pytest.fixture(scope="module")
def cli(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("evaluation-cli")
    source = directory / "harness.c"
    binary = directory / "evaluation-cli"
    source.write_text(HARNESS, encoding="utf-8")
    subprocess.run(
        [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
            str(ROOT / "src/evaluate.c"),
            str(ROOT / "src/benchmark.c"),
            str(ROOT / "src/scoring.c"),
            str(source),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
    )
    return binary


def run_eval(
    cli: Path,
    data: Path,
    scores: Path | None = None,
    env: dict[str, str] | None = None,
    context: int = 32,
):
    command = [
        str(cli),
        "eval",
        "--model",
        "stub.bin",
        "--data",
        str(data),
        "--kv",
        "f32",
        "--context",
        str(context),
        "--chunk",
        "3",
        "--threads",
        "1",
    ]
    if scores is not None:
        command.extend(("--scores", str(scores)))
    return subprocess.run(command, text=True, capture_output=True, env=env, check=False)


def test_lm_scores_exact_masks_counts_and_prefix(cli: Path, tmp_path: Path) -> None:
    data = tmp_path / "lm.smeval"
    data.write_bytes(
        smeval(
            1,
            [
                record(tokens=(0, 3, 2, 1), score_start=1, score_end=4),
                record(tokens=(2, 0, 3, 0), score_start=2, score_end=4),
            ],
        )
    )
    scores = tmp_path / "lm-scores.jsonl"
    completed = run_eval(cli, data, scores)
    assert completed.returncode == 0, completed.stderr
    metric = json.loads(completed.stdout)
    log_normalizer = math.log(sum(math.exp(value) for value in range(4)))
    expected_nll = (3 * log_normalizer - 6) + (2 * log_normalizer - 3)
    assert metric["kind"] == "lm"
    assert metric["record_count"] == 2
    assert metric["target_count"] == 5
    assert metric["input_token_count"] == 8
    assert metric["nll_sum"] == pytest.approx(expected_nll, abs=1e-14)
    assert metric["mean_nll"] == pytest.approx(expected_nll / 5, abs=1e-14)
    assert metric["perplexity"] == pytest.approx(math.exp(expected_nll / 5), abs=1e-14)
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    assert [row["target_count"] for row in rows] == [3, 2]
    assert rows[0]["normalized_loglikelihood"] == pytest.approx(rows[0]["loglikelihood"] / 3)


def test_mc_raw_and_unicode_character_normalized_accuracy(cli: Path, tmp_path: Path) -> None:
    data = tmp_path / "mc.smeval"
    data.write_bytes(
        smeval(
            2,
            [
                # norm_chars=2 represents two Unicode code points (for example "é界"), not five UTF-8 bytes.
                record(group=0, label=0, choice=0, nchoices=2, tokens=(0, 3, 3), score_start=1, score_end=3, norm_chars=2),
                record(group=0, label=0, choice=1, nchoices=2, tokens=(0, 0), score_start=1, score_end=2, norm_chars=10),
                record(group=1, label=1, choice=0, nchoices=2, tokens=(0, 0), score_start=1, score_end=2, norm_chars=1),
                record(group=1, label=1, choice=1, nchoices=2, tokens=(0, 3), score_start=1, score_end=2, norm_chars=1),
            ],
        )
    )
    scores = tmp_path / "mc-scores.jsonl"
    completed = run_eval(cli, data, scores)
    assert completed.returncode == 0, completed.stderr
    metric = json.loads(completed.stdout)
    assert metric["kind"] == "multiple_choice"
    assert metric["record_count"] == 4
    assert metric["example_count"] == 2
    assert metric["target_count"] == 5
    assert metric["norm_char_count"] == 14
    assert metric["raw_correct"] == 2
    assert metric["raw_accuracy"] == 1.0
    assert metric["normalized_correct"] == 1
    assert metric["normalized_accuracy"] == 0.5
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["raw_selected"] == 0
    assert rows[0]["normalized_selected"] == 1
    assert rows[0]["normalized_loglikelihood"] == pytest.approx(rows[0]["loglikelihood"] / 2)


def malformed_payloads() -> list[tuple[str, bytes]]:
    valid_lm = record(tokens=(0, 1), score_start=1, score_end=2)
    return [
        ("magic", struct.pack("<8sII", b"BADMAGIC", 1, 1) + valid_lm),
        ("trailing", smeval(1, [valid_lm]) + b"x"),
        ("truncated", smeval(1, [valid_lm])[:-1]),
        ("token", smeval(1, [record(tokens=(0, 4), score_start=1, score_end=2)])),
        ("bounds", smeval(1, [record(tokens=(0, 1), score_start=0, score_end=2)])),
        ("lm_metadata", smeval(1, [record(tokens=(0, 1), score_start=1, score_end=2, norm_chars=1)])),
        ("mc_incomplete", smeval(2, [record(group=0, label=0, choice=0, nchoices=2, tokens=(0, 1), score_start=1, score_end=2, norm_chars=1)])),
        ("mc_choice_order", smeval(2, [record(group=0, label=0, choice=1, nchoices=2, tokens=(0, 1), score_start=1, score_end=2, norm_chars=1), record(group=0, label=0, choice=0, nchoices=2, tokens=(0, 1), score_start=1, score_end=2, norm_chars=1)])),
    ]


@pytest.mark.parametrize(("name", "payload"), malformed_payloads(), ids=lambda value: value if isinstance(value, str) else None)
def test_malformed_inputs_fail_without_metrics_or_scores(
    cli: Path, tmp_path: Path, name: str, payload: bytes
) -> None:
    data = tmp_path / f"{name}.smeval"
    scores = tmp_path / f"{name}.jsonl"
    data.write_bytes(payload)
    completed = run_eval(cli, data, scores)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert completed.stderr
    assert not scores.exists()
    assert not list(tmp_path.glob(f"{name}.jsonl.tmp.*"))


def test_execution_failure_has_no_partial_success_output(cli: Path, tmp_path: Path) -> None:
    data = tmp_path / "failure.smeval"
    scores = tmp_path / "failure.jsonl"
    data.write_bytes(smeval(1, [record(tokens=(0, 1), score_start=1, score_end=2)]))
    environment = dict(os.environ, SM_STUB_FAIL="1")
    completed = run_eval(cli, data, scores, environment)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "injected execution failure" in completed.stderr
    assert not scores.exists()


@pytest.mark.parametrize(
    "name", ("wikitext2_smoke.smeval", "hellaswag_smoke.smeval", "piqa_smoke.smeval")
)
def test_current_prepared_smoke_metadata_passes_c_validation(cli: Path, name: str) -> None:
    data = ROOT / "data" / name
    if not data.is_file():
        pytest.skip("ignored prepared data is unavailable")
    environment = dict(os.environ, SM_STUB_FAIL="1", SM_STUB_VOCAB="49152")
    completed = run_eval(cli, data, env=environment, context=2048)
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "injected execution failure" in completed.stderr


def test_benchmark_defaults_and_medians(cli: Path) -> None:
    completed = subprocess.run(
        [
            str(cli),
            "bench",
            "--model",
            "stub.bin",
            "--kv",
            "f32",
            "--context",
            "256",
            "--chunk",
            "128",
            "--threads",
            "1",
            "--prompt",
            "128",
            "--decode",
            "128",
            "--warmup",
            "2",
            "--repetitions",
            "5",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metric = json.loads(completed.stdout)
    assert metric["prompt_tokens"] == 128
    assert metric["decode_tokens"] == 128
    assert metric["warmup_repetitions"] == 2
    assert metric["measured_repetitions"] == 5
    assert len(metric["prefill_seconds"]) == 5
    assert len(metric["decode_seconds"]) == 5
    assert metric["median_ttft_seconds"] == metric["median_prefill_seconds"]
    assert metric["prefill_tokens_per_second"] == pytest.approx(128 / 0.01)
    assert metric["decode_tokens_per_second"] == pytest.approx(128 / 0.01)


def test_benchmark_rejects_insufficient_context_without_metrics(cli: Path) -> None:
    completed = subprocess.run(
        [
            str(cli),
            "bench",
            "--model",
            "stub.bin",
            "--context",
            "128",
            "--prompt",
            "128",
            "--decode",
            "128",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stdout == ""


def test_matrix_runner_resumes_only_verified_identical_cells(tmp_path: Path) -> None:
    fake = tmp_path / "fake-smollm"
    fake.write_text(
        """#!/usr/bin/env python3
import fcntl, json, os, pathlib, sys, time
args=sys.argv[1:]
def option(name): return args[args.index(name)+1]
def active(delta):
    path=os.environ.get('FAKE_STATE')
    if not path: return
    with open(path,'a+') as state:
        fcntl.flock(state,fcntl.LOCK_EX); state.seek(0); text=state.read()
        value=json.loads(text) if text else {'active':0,'maximum':0}
        value['active']+=delta; value['maximum']=max(value['maximum'],value['active'])
        state.seek(0); state.truncate(); json.dump(value,state); state.flush()
        fcntl.flock(state,fcntl.LOCK_UN)
active(1); time.sleep(float(os.environ.get('FAKE_DELAY','0')))
score=pathlib.Path(option('--scores')); score.parent.mkdir(parents=True,exist_ok=True)
score.write_text(json.dumps({'model':option('--model'),'kv':option('--kv')})+'\\n')
with open(os.environ['FAKE_COUNTER'],'a') as out: out.write('run\\n')
active(-1)
print(json.dumps({'command':'eval','record_count':1,'target_count':1,'compiler':'fake'}))
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    models = tmp_path / "models"
    manifests = tmp_path / "manifests"
    data = tmp_path / "data"
    output = tmp_path / "results"
    models.mkdir()
    manifests.mkdir()
    data.mkdir()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for dtype in ("f32", "q8"):
        model = models / f"model-{dtype}.bin"
        model.write_bytes(dtype.encode())
        (manifests / f"model-{dtype}.json").write_text(
            json.dumps(
                {
                    "output": {"sha256": digest(model)},
                    "source": {"revision": "revision"},
                }
            ),
            encoding="utf-8",
        )
    dataset = data / "wikitext2_smoke.smeval"
    dataset.write_bytes(b"fixture")
    (manifests / "data_wikitext2_smoke.json").write_text(
        json.dumps(
            {
                "artifact": {"sha256": digest(dataset)},
                "source_revision": "data-revision",
            }
        ),
        encoding="utf-8",
    )
    counter = tmp_path / "counter"
    command = [
        "python3",
        str(ROOT / "scripts/run_benchmarks.py"),
        "eval",
        "--binary",
        str(fake),
        "--f32-model",
        str(models / "model-f32.bin"),
        "--q8-model",
        str(models / "model-q8.bin"),
        "--f32-manifest",
        str(manifests / "model-f32.json"),
        "--q8-manifest",
        str(manifests / "model-q8.json"),
        "--data-dir",
        str(data),
        "--manifest-dir",
        str(manifests),
        "--output-dir",
        str(output),
        "--tasks",
        "wikitext2",
        "--variant",
        "smoke",
        "--context",
        "32",
        "--progress-every",
        "0",
        "--jobs",
        "2",
    ]
    state = tmp_path / "state.json"
    environment = dict(
        os.environ,
        FAKE_COUNTER=str(counter),
        FAKE_STATE=str(state),
        FAKE_DELAY="0.1",
    )
    first = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    assert first.returncode == 0, first.stderr
    assert counter.read_text(encoding="utf-8").count("run") == 4
    assert json.loads(state.read_text(encoding="utf-8"))["maximum"] >= 2
    rows = [json.loads(line) for line in (output / "evaluation-smoke.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert [row["configuration"] for row in rows] == [
        "fp32_fp32",
        "w8a8_fp32",
        "fp32_kv8",
        "w8a8_kv8",
    ]
    assert all(row["result_sha256"] and row["scores_sha256"] for row in rows)
    assert all(row["evaluation_concurrency"] == 2 for row in rows)
    assert all(row["timing_isolated"] is False for row in rows)
    assert all(row["timing_scope"] == "concurrent_not_isolated" for row in rows)

    resumed = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    assert resumed.returncode == 0, resumed.stderr
    assert counter.read_text(encoding="utf-8").count("run") == 4

    score = Path(rows[0]["scores_path"])
    score.write_text("tampered\n", encoding="utf-8")
    rejected = subprocess.run(command, text=True, capture_output=True, env=environment, check=False)
    assert rejected.returncode != 0
    assert "SHA256 mismatch" in rejected.stderr
    assert counter.read_text(encoding="utf-8").count("run") == 4

    restarted = subprocess.run(
        [*command, "--restart"], text=True, capture_output=True, env=environment, check=False
    )
    assert restarted.returncode == 0, restarted.stderr
    assert counter.read_text(encoding="utf-8").count("run") == 8

    selected = subprocess.run(
        [
            *command,
            "--restart",
            "--output-stem",
            "selected",
            "--configurations",
            "fp32_kv8",
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )
    assert selected.returncode == 0, selected.stderr
    selected_rows = [
        json.loads(line) for line in (output / "selected.jsonl").read_text().splitlines()
    ]
    assert [row["configuration"] for row in selected_rows] == ["fp32_kv8"]
    assert counter.read_text(encoding="utf-8").count("run") == 9

    malformed_jobs = subprocess.run(
        [*command, "--jobs", "0"], text=True, capture_output=True, env=environment, check=False
    )
    assert malformed_jobs.returncode == 2
    assert "positive" in malformed_jobs.stderr

    bench_jobs = subprocess.run(
        ["python3", str(ROOT / "scripts/run_benchmarks.py"), "bench", "--jobs", "2"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert bench_jobs.returncode == 2
    assert "unrecognized arguments" in bench_jobs.stderr

    bench_selection = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/run_benchmarks.py"),
            "bench",
            "--binary",
            str(fake),
            "--f32-model",
            str(models / "model-f32.bin"),
            "--q8-model",
            str(models / "model-q8.bin"),
            "--f32-manifest",
            str(manifests / "model-f32.json"),
            "--q8-manifest",
            str(manifests / "model-q8.json"),
            "--output-dir",
            str(output),
            "--configurations",
            "w8a8_kv8",
            "--prompts",
            "128",
            "--thread-counts",
            "1",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert bench_selection.returncode == 0, bench_selection.stderr
    bench_plan = [json.loads(line) for line in bench_selection.stdout.splitlines()]
    assert len(bench_plan) == 1
    assert bench_plan[0]["job_id"] == "w8a8_kv8:prompt128:threads1"
