"""The reporter a connected repo's OWN CI runs. Copy this file into your
repo (e.g. .github/scripts/report_scan.py) and call it from a workflow
step -- see templates/sera_report_scan.yml for the full example.

This script has ONE job: run your own canary entrypoint (declared in your
own .sera.yml, running with your own dependencies in your own CI
environment) and POST the result to SERA's dashboard. It never sends
your code, only the resulting accuracy/confidence numbers.

On a regression, it also runs the controlled-intervention sequence: each
of the top few ranked candidate commits is reverted and re-measured
INDIVIDUALLY (never all at once first), and only if none alone restores
the baseline does it try the COMBINED revert of every candidate that
showed real individual improvement. All of this still runs entirely in
the tenant's own CI (same checkout, same dependencies) -- it is not SERA
executing anything on its own infrastructure.

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


def current_head_sha() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return proc.stdout.strip() if proc.returncode == 0 else None


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


def revert_and_measure(entrypoint: str, file_commit_pairs: list[tuple[str, str]]) -> tuple[list, float] | None:
    """Reverts each (path, commit_sha) pair to that commit's PARENT
    version, re-runs the canary once, and always restores the working
    tree afterward -- regardless of success or failure. A pair list can
    span multiple different commits (the combined-revert case), each
    file reverted relative to its OWN commit's parent.

    Returns (files_actually_reverted, after_metric), or None if none of
    the files could be reverted at all (e.g. shallow clone with no
    parent commit) -- that degrades to "no attempt recorded", not a
    failed report.
    """
    reverted = []
    all_paths = [p for p, _ in file_commit_pairs]
    try:
        for path, commit_sha in file_commit_pairs:
            proc = subprocess.run(["git", "show", f"{commit_sha}^:{path}"], capture_output=True, timeout=60)
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
        subprocess.run(["git", "checkout", "--", *all_paths], capture_output=True)


def run_intervention_sequence(entrypoint: str, candidates: list[dict], before_metric: float) -> list[dict]:
    """Tries each candidate INDIVIDUALLY first (never all at once). Stops
    as soon as one alone restores the baseline (checked server-side; here
    we just keep going through all candidates regardless, since we don't
    know the regression threshold client-side -- an extra attempt or two
    costs a few seconds, not correctness). If nothing tested individually
    fully explains it, tries the COMBINED revert of every candidate that
    showed *some* real improvement on its own -- never blindly all of
    them, and never ones that showed zero individual improvement.
    """
    attempts = []
    contributing_pairs: list[tuple[str, str]] = []  # (path, commit_sha) for the combined attempt

    for cand in candidates:
        sha = cand.get("commit_sha")
        files = cand.get("files_changed") or []
        if not sha or not files:
            continue
        outcome = revert_and_measure(entrypoint, [(f, sha) for f in files])
        if outcome is None:
            continue
        reverted, after_metric = outcome
        print(f"  [{sha[:10]}] reverted {reverted} -> metric={after_metric}")
        attempts.append(
            {
                "commit_sha": sha,
                "stage_label": cand.get("stage_label"),
                "files_reverted": reverted,
                "after_metric": after_metric,
                "combined": False,
            }
        )
        if after_metric > before_metric + 0.01:
            contributing_pairs.extend((f, sha) for f in reverted)

    if len(contributing_pairs) > 1:
        # More than one candidate individually helped but none alone was
        # enough -- try restoring all of them together in one combined
        # measurement. This is the only place multiple commits' files are
        # ever reverted simultaneously, and only because each was already
        # shown, individually, to move the metric in the right direction.
        outcome = revert_and_measure(entrypoint, contributing_pairs)
        if outcome is not None:
            reverted, after_metric = outcome
            print(f"  [combined: {sorted({s for _, s in contributing_pairs})}] reverted {reverted} -> metric={after_metric}")
            attempts.append(
                {
                    "commit_sha": ",".join(sorted({s for _, s in contributing_pairs})),
                    "stage_label": "combined",
                    "files_reverted": reverted,
                    "after_metric": after_metric,
                    "combined": True,
                }
            )

    return attempts


def report_validation_sequence(dashboard_url: str, repo_id: str, token: str, scan_id: int, interventions: list[dict]) -> dict:
    body = json.dumps(
        {"report_token": token, "scan_id": scan_id, "interventions": interventions}
    ).encode()
    req = urllib.request.Request(
        f"{dashboard_url.rstrip('/')}/report/{repo_id}/validate-scan-sequence",
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

    head_sha = current_head_sha()
    if head_sha:
        result["head_sha"] = head_sha

    response = report(dashboard_url, repo_id, token, result)
    print(f"reported to SERA: {response}")

    if response.get("status") == "incident":
        print("::warning::SERA detected a regression -- check your dashboard for the ranked cause.")

        candidates = response.get("candidates") or (
            [response["top_candidate"]] if response.get("top_candidate") else []
        )
        if candidates:
            before_metric = _extract_metric(result)
            print(f"running controlled-intervention sequence over {len(candidates)} candidate(s)...")
            attempts = run_intervention_sequence(entrypoint, candidates, before_metric)
            if not attempts:
                print("intervention skipped -- couldn't revert any candidate's files (e.g. shallow checkout)")
            else:
                validation = report_validation_sequence(dashboard_url, repo_id, token, response["scan_id"], attempts)
                print(f"reported validation sequence to SERA: {validation}")
                verdicts = [a["verdict"] for a in validation.get("interventions", [])]
                if any(v in ("validated", "validated_combined") for v in verdicts):
                    print("::notice::Intervention confirmed a cause -- accuracy recovered after reverting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
