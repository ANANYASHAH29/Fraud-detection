"""The reporter a connected repo's OWN CI runs. Copy this file into your
repo (e.g. .github/scripts/report_scan.py) and call it from a workflow
step -- see templates/sera_report_scan.yml for the full example.

This script has ONE job: run your own canary entrypoint (declared in your
own .sera.yml, running with your own dependencies in your own CI
environment) and POST the result to SERA's dashboard. It never sends
your code, only the resulting accuracy/confidence numbers.

Required environment variables (set as repo secrets, never checked in):
    SERA_DASHBOARD_URL   e.g. https://your-sera-instance.example.com
    SERA_REPO_ID         the numeric id from your SERA dashboard's Settings page
    SERA_REPORT_TOKEN    the token from that same Settings page

Usage:
    python scripts/report_scan.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

import yaml


def load_canary_entrypoint() -> str:
    with open(".sera.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    entrypoint = (config.get("canary") or {}).get("entrypoint")
    if not entrypoint:
        raise RuntimeError(".sera.yml has no canary.entrypoint set")
    return entrypoint


def run_canary(entrypoint: str) -> dict:
    command = ["python", entrypoint] if entrypoint.endswith(".py") else [entrypoint]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"canary entrypoint failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("canary entrypoint produced no stdout")
    return json.loads(lines[-1])


def report(dashboard_url: str, repo_id: str, token: str, result: dict) -> dict:
    # Two schemas share this one endpoint contract: the CV-style one
    # (accuracy/mean_confidence/...) posts to /scan, the generic
    # task-agnostic one (task_type/primary_metric/...) posts to
    # /generic-scan. Auto-detect by which keys are actually present so
    # this script works unmodified for either kind of canary_eval.py.
    path = "generic-scan" if "task_type" in result else "scan"
    body = json.dumps({**result, "report_token": token}).encode()
    req = urllib.request.Request(
        f"{dashboard_url.rstrip('/')}/report/{repo_id}/{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main():
    dashboard_url = os.environ.get("SERA_DASHBOARD_URL")
    repo_id = os.environ.get("SERA_REPO_ID")
    token = os.environ.get("SERA_REPORT_TOKEN")
    if not all([dashboard_url, repo_id, token]):
        print(
            "SERA_DASHBOARD_URL, SERA_REPO_ID, and SERA_REPORT_TOKEN must all be set "
            "(as repo secrets/variables) -- see this file's docstring.",
            file=sys.stderr,
        )
        return 1

    entrypoint = load_canary_entrypoint()
    print(f"running canary entrypoint: {entrypoint}")
    result = run_canary(entrypoint)
    if "task_type" in result:
        print(f"{result.get('primary_metric_name')}={result.get('primary_metric')}")
    else:
        print(f"accuracy={result.get('accuracy')}  mean_confidence={result.get('mean_confidence')}")

    response = report(dashboard_url, repo_id, token, result)
    print(f"reported to SERA: {response}")

    if response.get("status") == "incident":
        print("::warning::SERA detected a regression -- check your dashboard for the ranked cause.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
