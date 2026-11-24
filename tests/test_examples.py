"""The demo scripts run end to end on synthetic data."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_demo_judge_checks():
    out = run("examples/demo_judge_checks.py")
    assert "oracle    USABLE" in out and "position  NOT USABLE" in out and "length    NOT USABLE" in out


def test_demo_eval_methods():
    out = run("examples/demo_eval_methods.py", "--boot", "60", "--n", "300")
    assert "SYNTHETIC DATA" in out and "predictive band" in out and "RECALIBRATE" in out
