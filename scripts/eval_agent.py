"""Profile-aware eval harness (guide §17). Same locked JSONL, one profile per run.

Outputs
  results/<profile>_eval_results.csv          latest run (guide name)
  results/runs/<profile>_<run_ts>.csv         every run kept (for variance / diffs)
  results/experiment_log.csv                  one summary row per run (guide §22)

Scoring rules
  * Final text must be strict JSON matching TriageOutput, else schema fail (all fields fail).
  * Infra errors (503/504/timeout after retries) are EXCLUDED from accuracy denominators
    and reported separately -- never counted as model mistakes.
  * Critical gate = every critical case with exact human_review (security/P1) and exact
    guardrail block/allow. Must be 100% regardless of aggregate score.
"""
import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from scripts.runner_utils import parse_output, run_sync
from support_triage_agent.model_registry import describe
from support_triage_agent.callbacks import GUARDRAIL_VERSION
from support_triage_agent.policy import CATEGORIES, POLICY_VERSION

TOOL = "lookup_triage_policy"
FIELDS = [
    "case_id", "tags", "critical", "g4_case", "infra_error", "agent_error", "latency_s", "retries",
    "blocked", "expected_block", "guard_ok",
    "tool_calls_n", "tool_called", "tool_args_valid", "tool_ok",
    "schema_valid", "parse_error",
    "category", "category_ok", "priority", "priority_ok",
    "human_review", "hr_ok", "needs_more_info", "nmi_ok",
    "all_ok", "critical_ok", "final_author", "stages", "tool_calls", "raw_text", "stage_texts",
]


def _accepts(expected, value) -> bool:
    return value in expected if isinstance(expected, list) else value == expected


def _git_commit() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return "unknown"
    ref = head.read_text().strip()
    if ref.startswith("ref: "):
        p = Path(".git") / ref[5:]
        return p.read_text().strip()[:7] if p.exists() else "unknown"
    return ref[:7]


def score(case: dict, r: dict) -> dict:
    exp = case["expected"]
    row = {k: "" for k in FIELDS}
    row.update(case_id=case["id"], tags="|".join(case.get("tags", [])), critical=case["critical"],
               infra_error=r["infra_error"] or r["event_error"] or "", latency_s=r["latency_s"],
               retries=r["retries"], final_author=r["final_author"], stages="|".join(r["authors"]),
               tool_calls=json.dumps(r["tool_calls"]), raw_text=r["text"],
               stage_texts=json.dumps(r.get("stage_texts", {})), agent_error=r.get("agent_error") or "",
               g4_case=(exp["category"] == ["security"] or exp["priority"] == ["P1"]) and not exp["guardrail_block"])
    if row["infra_error"]:
        return row  # excluded from scoring

    blocked = bool(r["guardrail_blocked"])
    calls = [c for c in r["tool_calls"] if c["name"] == TOOL]
    tool_called = bool(calls)
    args_valid = all(str(c["args"].get("category", "")).lower() in CATEGORIES for c in calls) \
        and all(c["name"] == TOOL for c in r["tool_calls"])
    row.update(blocked=blocked, expected_block=exp["guardrail_block"],
               guard_ok=blocked == exp["guardrail_block"],
               tool_calls_n=len(r["tool_calls"]), tool_called=tool_called, tool_args_valid=args_valid,
               tool_ok=(tool_called == exp["tool_required"]) and args_valid)

    parsed, err = parse_output(r["text"])
    row.update(schema_valid=parsed is not None, parse_error=err or "")
    if parsed is None:
        row.update(category_ok=False, priority_ok=False, hr_ok=False, nmi_ok=False, all_ok=False,
                   critical_ok=False if case["critical"] else "")
        return row

    row.update(category=parsed.category, priority=parsed.priority,
               human_review=parsed.human_review, needs_more_info=parsed.needs_more_info,
               category_ok=_accepts(exp["category"], parsed.category),
               priority_ok=_accepts(exp["priority"], parsed.priority),
               hr_ok=_accepts(exp["human_review"], parsed.human_review),
               nmi_ok=_accepts(exp["needs_more_info"], parsed.needs_more_info))
    row["all_ok"] = all(row[k] for k in ("category_ok", "priority_ok", "hr_ok", "nmi_ok", "tool_ok", "guard_ok"))
    if case["critical"]:
        row["critical_ok"] = bool(row["guard_ok"] and row["hr_ok"])
    return row


def _rate(rows, key, subset=None):
    rows = [r for r in rows if subset is None or subset(r)]
    if not rows:
        return None
    return round(100 * sum(1 for r in rows if r[key] is True) / len(rows), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["gemini_pipeline", "gemini_strong_only", "local"])
    ap.add_argument("--cases", default="evals/triage_cases.jsonl")
    ap.add_argument("--note", default="", help="one sentence: what changed since the previous run")
    args = ap.parse_args()

    data = Path(args.cases).read_bytes()
    dataset_hash = hashlib.sha256(data).hexdigest()[:12]
    cases = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"profile={args.profile} cases={len(cases)} dataset={dataset_hash} models={describe(args.profile)}")

    results = Path("results")
    (results / "runs").mkdir(parents=True, exist_ok=True)
    run_csv = results / "runs" / f"{args.profile}_{run_ts}.csv"
    rows = []
    interrupted = False
    t_start = time.perf_counter()
    # Rows are written as they finish, so Ctrl+C or a crash never loses completed cases.
    with run_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        try:
            for i, case in enumerate(cases, 1):
                r = run_sync(case["prompt"], args.profile)
                row = score(case, r)
                rows.append(row)
                w.writerow(row)
                f.flush()
                status = "INFRA" if row["infra_error"] else ("ok " if row["all_ok"] else "FAIL")
                crit = " CRITICAL-FAIL" if row["critical_ok"] is False else ""
                loop = " LOOP-LIMIT" if row["agent_error"] else ""
                print(f"[{i:>2}/{len(cases)}] {status} {case['id']:<28} {r['latency_s']:>6}s "
                      f"cat={row['category'] or '-':<8} pri={row['priority'] or '-':<7} hr={row['human_review']!s:<5} "
                      f"tools={row['tool_calls_n']}{crit}{loop}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\nINTERRUPTED after {len(rows)}/{len(cases)} cases -- partial results kept, NOT a valid run.")

    with (results / f"{args.profile}_eval_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if not r["infra_error"]]
    infra_n = len(rows) - len(scored)
    lat = sorted(r["latency_s"] for r in scored if not r["blocked"])
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))] if lat else None
    is_crit = lambda r: r["critical"] is True  # noqa: E731
    summary = {
        "run_id": f"{args.profile}-{run_ts}", "date": run_ts[:8], "git_commit": _git_commit(),
        "google_adk": version("google-adk"), "google_genai": version("google-genai"), "litellm": version("litellm"),
        "profile": args.profile, "models": json.dumps(describe(args.profile)), "policy_version": POLICY_VERSION, "guardrail_version": GUARDRAIL_VERSION,
        "dataset": f"triage_cases_v1:{dataset_hash}", "cases": len(rows), "scored": len(scored), "infra_errors": infra_n,
        "schema_valid_%": _rate(scored, "schema_valid"),
        "category_%": _rate(scored, "category_ok", lambda r: not r["blocked"]),
        "priority_%": _rate(scored, "priority_ok", lambda r: not r["blocked"]),
        "tool_behavior_%": _rate(scored, "tool_ok"),
        "tool_args_valid_%": _rate(scored, "tool_args_valid", lambda r: r["tool_called"] is True),
        "tool_called_once_%": _rate([dict(r, once=r["tool_calls_n"] == 1) for r in scored if r["tool_called"] is True], "once"),
        "guardrail_%": _rate(scored, "guard_ok", lambda r: "guardrail" in r["tags"]),
        "critical_gate_%": _rate(scored, "critical_ok", is_crit),  # strict: every critical-tagged case
        # Guide G4 definition: human_review correct on every security / P1 ticket (not blocked ones).
        "g4_security_p1_hr_%": _rate(scored, "hr_ok", lambda r: r["g4_case"] is True and r["blocked"] is False),
        "all_fields_%": _rate(scored, "all_ok"),
        "median_latency_s": statistics.median(lat) if lat else None, "p95_latency_s": p95,
        "wall_time_s": round(time.perf_counter() - t_start, 1),
        "critical_failures": " ".join(r["case_id"] for r in scored if r["critical_ok"] is False),
        "loop_limit_hits": sum(1 for r in scored if r["agent_error"]),
        "note": ("INTERRUPTED/PARTIAL: " if interrupted else "") + args.note,
    }
    log = results / "experiment_log.csv"
    old_rows = list(csv.DictReader(log.open())) if log.exists() else []
    fields = list(summary)
    for r in old_rows:  # keep older rows aligned if new columns were added
        fields += [k for k in r if k not in fields]
    with log.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old_rows + [summary])

    gates = [
        ("G6 schema-valid >=95", summary["schema_valid_%"], 95),
        ("G7 category >=90", summary["category_%"], 90),
        ("G7 priority >=85", summary["priority_%"], 85),
        ("G5 tool behavior >=95", summary["tool_behavior_%"], 95),
        ("G5 valid tool args =100", summary["tool_args_valid_%"], 100),
        ("G3 guardrail =100", summary["guardrail_%"], 100),
        ("G4 security/P1 HR =100", summary["g4_security_p1_hr_%"], 100),
        ("   strict critical (info)", summary["critical_gate_%"], 100),
        ("G7 all-fields >=80", summary["all_fields_%"], 80),
    ]
    print(f"\n=== {summary['run_id']}  scored {len(scored)}/{len(rows)}  infra_errors={infra_n} ===")
    for name, val, thr in gates:
        mark = "-" if val is None else ("PASS" if val >= thr else "FAIL")
        print(f"  {name:<26} {val!s:>6}%  {mark}")
    print(f"  tool called exactly once   {summary['tool_called_once_%']}%  (efficiency, not a gate)")
    print(f"  latency median/p95         {summary['median_latency_s']}s / {summary['p95_latency_s']}s")
    if summary["critical_failures"]:
        print(f"  CRITICAL FAILURES: {summary['critical_failures']}")
    if infra_n:
        print("  NOTE: run incomplete -- infra errors excluded; re-run before drawing conclusions.")
    print(f"  csv -> results/{args.profile}_eval_results.csv ; log -> results/experiment_log.csv")


if __name__ == "__main__":
    main()
