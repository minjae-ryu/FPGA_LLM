"""A mistyped experimental option must never silently select a default."""
from pathlib import Path
import subprocess
import pytest

BIN = Path(__file__).resolve().parents[1] / "build/smollm"

@pytest.mark.parametrize("arguments,message", [
    (["inspect","--model","unused","--threds","12"],"unknown option"),
    (["logits","--model","unused","--chunk","1","--chunk","128"],"duplicate option"),
    (["bench","--model","unused","--warmup"],"missing value"),
    (["eval","--model","unused","--report-only"],"unknown option"),
])
def test_invalid_experiment_options(arguments, message):
    completed=subprocess.run([str(BIN),*arguments],capture_output=True,text=True)
    assert completed.returncode != 0
    assert message in completed.stderr
    assert not completed.stdout
