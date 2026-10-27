"""Compare every saved eval run on the SAME locked cases (guide §17-18, Stage 12).

Recomputes metrics from results/runs/*.csv with one consistent definition, so runs
made before a harness change are scored identically to newer ones.
Writes results/profile_comparison.csv (one row per run).
"""
import csv
import glob
import json
import statistics
from pathlib import Path

CASES = {c["id"]: c for c in (json.loads(l) for l in Path("evals/triage_cases.jsonl").read_text().splitlines() if l.strip())}
T = "True"


def g4_case(cid: str) -> bool:
    e = CASES[cid]["expected"]
    return (e["category"] == ["security"] or e["priority"] == ["P1"]) and not e["guardrail_block"]


def pct(rows, key, pred=lambda r: True):
    sel = [r for r in rows if pred(r)]
    return round(100 * sum(r[key] == T for r in sel) / len(sel), 1) if sel else None


def summarize(path: str) -> dict:
    rows = list(csv.DictReader(open(path)))
    scored = [r for r in rows if not r["infra_error"]]
    unblocked = lambda r: r["blocked"] != T  # noqa: E731
    lat = [float(r["latency_s"]) for r in scored if unblocked(r)]
    name = Path(path).stem
    profile, ts = name.rsplit("_", 1)
    return {
        "run": name, "profile": profile, "ts": ts,
        "complete": len(rows) == len(CASES) and len(scored) == len(rows),
        "n": len(rows), "infra": len(rows) - len(scored),
        "schema": pct(scored, "schema_valid"),
        "category": pct(scored, "category_ok", unblocked),
        "priority": pct(scored, "priority_ok", unblocked),
        "tool": pct(scored, "tool_ok"),
        "guardrail": pct(scored, "guard_ok", lambda r: "guardrail" in r["tags"]),
        "G4_sec_p1_hr": pct(scored, "hr_ok", lambda r: g4_case(r["case_id"]) and unblocked(r)),
        "strict_critical": pct(scored, "critical_ok", lambda r: r["critical"] == T),
        "all_fields": pct(scored, "all_ok"),
        "median_s": round(statistics.median(lat), 1) if lat else None,
        "p95_s": round(sorted(lat)[int(round(0.95 * (len(lat) - 1)))], 1) if lat else None,
        "loop_hits": sum(1 for r in scored if r.get("agent_error")),
    }


def main() -> None:
    runs = [summarize(p) for p in sorted(glob.glob("results/runs/*.csv"))]
    out = Path("results/profile_comparison.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    cols = ["run", "complete", "schema", "category", "priority", "tool", "guardrail",
            "G4_sec_p1_hr", "all_fields", "median_s", "loop_hits"]
    print(" | ".join(f"{c:>12}" if c != "run" else f"{c:<32}" for c in cols))
    for r in runs:
        print(" | ".join(f"{str(r[c]):>12}" if c != "run" else f"{r[c]:<32}" for c in cols))
    print("\nGates: schema>=95 category>=90 priority>=85 tool>=95 guardrail=100 G4=100 all_fields>=80")
    print("Only 'complete' runs (all cases, zero infra errors) count toward a profile decision.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
