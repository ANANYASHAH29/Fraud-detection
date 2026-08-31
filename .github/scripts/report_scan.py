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


def _extract_metric(result: dict) -> float:
    return result["primary_metric"] if "task_type" in result else result["accuracy"]


def run_intervention(entrypoint: str, commit_sha: str, files_changed: list) -> tuple[list, float] | None:
    """The controlled-intervention step: revert exactly the suspect
    commit's flagged files to their state at the PARENT of `commit_sha`
    (not necessarily HEAD -- the scheduled/drift-catching trigger can flag
    a commit that isn't the tip), re-run the same canary entrypoint, and
    report whether accuracy recovers. This still runs entirely in the
    tenant's own CI (same checkout, same dependencies) -- it is not SERA
    re-executing anything on its own infrastructure, just a second call to
    the same script the tenant already trusts to run.

    Returns (files_actually_reverted, after_metric), or None if the revert
    itself wasn't possible (e.g. shallow clone with no parent commit) --
    that degrades to "no validation available", not a failed report.
    """
    reverted = []
    try:
        for path in files_changed:
            proc = subprocess.run(
                ["git", "show", f"{commit_sha}^:{path}"], capture_output=True, timeout=60
            )
            if proc.returncode != 0:
                continue  # e.g. file didn't exist before this commit -- nothing to revert
            with open(path, "wb") as f:
                f.write(proc.stdout)
            reverted.append(path)

        if not reverted:
            return None

        after_result = run_canary(entrypoint)
        return reverted, _extract_metric(after_result)
    finally:
        # Always restore the working tree to what was actually committed,
        # regardless of success/failure above -- this runner's checkout
        # should never be left mid-revert.
        subprocess.run(["git", "checkout", "--", *files_changed], capture_output=True)


def report_validation(dashboard_url: str, repo_id: str, token: str, scan_id: int, files_reverted: list, after_metric: float) -> dict:
    body = json.dumps(
        {
            "report_token": token,
            "scan_id": scan_id,
            "files_reverted": files_reverted,
            "after_metric": after_metric,
        }
    ).encode()
    req = urllib.request.Request(
        f"{dashboard_url.rstrip('/')}/report/{repo_id}/validate-scan",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


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

        top_candidate = response.get("top_candidate")
        if top_candidate and top_candidate.get("files_changed") and top_candidate.get("commit_sha"):
            print(f"running controlled intervention on {top_candidate['files_changed']}...")
            outcome = run_intervention(
                entrypoint, top_candidate["commit_sha"], top_candidate["files_changed"]
            )
            if outcome is None:
                print("intervention skipped -- couldn't revert the suspect files (e.g. shallow checkout)")
            else:
                files_reverted, after_metric = outcome
                print(f"after reverting {files_reverted}: metric={after_metric}")
                validation = report_validation(
                    dashboard_url, repo_id, token, response["scan_id"], files_reverted, after_metric
                )
                print(f"reported validation to SERA: {validation}")
                if validation.get("recovered"):
                    print("::notice::Intervention confirmed the cause -- accuracy recovered after reverting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
