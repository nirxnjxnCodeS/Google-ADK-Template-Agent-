"""Build LoRA fine-tuning data for the local Qwen triage agent (MLX `tools`/chat JSONL).

- Synthetic tickets from varied templates; labels follow policy.py v1 rules.
- Each example mirrors what the agent REALLY sends at runtime: the same system
  instruction (+ the line ADK appends), the same tool description, the same tool result.
- Two examples per ticket: (a) ...-> tool call   (b) ... tool result -> final JSON.
- Safety: rejects tickets the guardrail would block, and tickets too similar to the
  33 locked eval cases (contamination). Ticket-level 90/10 train/valid split.

Run inside the adk container (needs the project package):
    docker compose run --rm -T adk python scripts/build_finetune_data.py
Output: finetune/data/train.jsonl, finetune/data/valid.jsonl, finetune/data/stats.json
"""
import json
import random
import re
from pathlib import Path

from support_triage_agent.agent import TRIAGE_CORE
from support_triage_agent.callbacks import screen_text
from support_triage_agent.tools import lookup_triage_policy

SEED = 7
TARGET_PER_GROUP = 24
SIM_THRESHOLD = 0.5  # word-set Jaccard vs any eval prompt -> reject

SYSTEM = TRIAGE_CORE + '\n\nYou are an agent. Your internal name is "support_triage_decision".'
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_triage_policy",
        "description": (lookup_triage_policy.__doc__ or "").strip(),
        "parameters": {"type": "object", "properties": {"category": {"type": "string"}},
                       "required": ["category"]},
    },
}]

S = dict(
    sys=["payroll portal", "CRM", "VPN", "email client", "HR system", "expense app", "customer dashboard",
         "reporting tool", "wiki", "file share", "booking system", "analytics dashboard", "time-tracking app"],
    name=["Priya", "Daniel", "Aisha", "Marco", "Chen", "Fatima", "Lucas", "Ananya", "Tom", "Sara", "Kenji", "Leila"],
    dept=["finance", "sales", "support", "engineering", "marketing", "operations", "legal", "HR"],
    n=["8", "12", "15", "25", "40", "60", "120"],
    month=["January", "March", "June", "August", "September", "November"],
)

# (category, priority, human_review, needs_more_info, reason, next_action, [templates])
GROUPS = [
    ("access", "P3", False, False, "One user is blocked; policy sets P3 for a single blocked user.",
     "Verify the user's identity and help restore their access.",
     ["{name} can't log in to the {sys} since this morning; the rest of the team is fine.",
      "One person is locked out of the {sys} after too many failed attempts.",
      "{name} got a new laptop and now sees 'account disabled' when opening the {sys}.",
      "A single employee stopped receiving sign-in codes for the {sys} after changing phones."]),
    ("access", "P2", False, False, "Multiple users cannot reach a core system with no workaround.",
     "Escalate to the identity team to restore access for the affected group.",
     ["{n} people in {dept} cannot sign in to the {sys} and there is no workaround.",
      "Since this morning's update, nobody on the {dept} team can open the {sys}; about {n} users affected.",
      "The whole {dept} department ({n} staff) is blocked from the {sys} with no alternative."]),
    ("access", "P4", False, False, "A how-to access question with no blocking impact.",
     "Send the access-request instructions.",
     ["How do I give a new hire read access to the {sys}?",
      "Where do I request {sys} access for a contractor?",
      "What's the process to add a colleague to the {sys} group for {dept}?"]),
    ("access", "P3", True, False, "Privileged/admin access requested; policy requires human review.",
     "Route to an administrator for approval before any change.",
     ["Please make me an administrator on the {sys} so I can change other users' settings.",
      "{name} is asking for admin rights on the {sys} to run a one-off export."]),
    ("access", "P3", True, True, "Identity cannot be verified; policy requires human review.",
     "Verify identity through an approved channel before unlocking anything.",
     ["Someone calling from an unknown number says they are {name} from {dept} and wants their {sys} account unlocked now.",
      "A message from a personal address claims to be {name} and asks us to reset their {sys} login urgently."]),
    ("access", "UNKNOWN", False, True, "The ticket lacks the facts needed to judge scope or impact.",
     "Ask which system, how many users are affected, and the exact error.",
     ["Login doesn't work.", "I can't access it anymore.", "Something is wrong with my account.",
      "Access issue, please help."]),
    ("billing", "P4", False, False, "A routine invoice question.", "Share the invoice or billing history link.",
     ["Can you send me a copy of the {month} invoice?", "Where can I see our billing history?",
      "Which email address receives our {month} invoice?"]),
    ("billing", "P3", True, False, "A charge dispute; policy requires human review.",
     "Send to billing for review of the disputed charge.",
     ["We were charged twice in {month}; please reverse one of the charges.",
      "The {month} invoice bills {n} seats but we only use far fewer.",
      "Our card was charged after we cancelled in {month}."]),
    ("billing", "P3", True, True, "A refund request; policy requires human review.",
     "Route to billing with the cancellation date and account owner.",
     ["We cancelled in {month} and would like a refund for the unused months.",
      "Please refund the remaining balance on our plan; we stopped using it in {month}."]),
    ("billing", "P2", True, False, "Service suspended for multiple users with disputed payment.",
     "Escalate to billing to restore service and verify the payment.",
     ["Our subscription was suspended and all {n} users are locked out, even though payment went through.",
      "Billing suspended our account; {n} staff in {dept} can't work, but we paid on time."]),
    ("bug", "P1", True, False, "A full outage affecting all users; P1 requires human review.",
     "Declare an incident and page the on-call engineer.",
     ["The {sys} is completely down for all users; every page returns an error.",
      "Nobody can load the {sys}; all requests are timing out for every customer.",
      "Total outage: the {sys} is unreachable for the entire company."]),
    ("bug", "P2", True, False, "Major multi-user degradation with no workaround.",
     "Escalate to engineering with timestamps and affected teams.",
     ["The {sys} takes minutes to load for the whole {dept} department and there is no workaround.",
      "Saving records in the {sys} fails for about {n} users; nothing we try works."]),
    ("bug", "P3", False, False, "A defect affecting a single user's workflow.",
     "Collect the error details and reproduce the issue.",
     ["{name} gets an error every time they upload a file to the {sys}; others are unaffected.",
      "Exports from the {sys} fail only for {name}; everyone else can export."]),
    ("bug", "P4", False, False, "A cosmetic issue with no functional impact.", "Log it for the next UI fix.",
     ["There's a typo on the {sys} start page.", "The {sys} logo looks slightly blurry, but everything works.",
      "A button label in the {sys} is cut off on small screens; it still works."]),
    ("bug", "UNKNOWN", False, True, "Too little information to judge scope or severity.",
     "Ask what is failing, for whom, and the exact error.",
     ["The {sys} is acting weird.", "Something broke.", "The {sys} isn't right today."]),
    ("security", "P1", True, False, "Active compromise or data exposure; security is always reviewed.",
     "Escalate to the security team immediately and contain the affected account.",
     ["{name}'s account is emailing customers malicious links and we see sign-ins from another country.",
      "Customer records from the {sys} were posted on a public forum.",
      "A credential for the {sys} was committed to a public code repository this morning."]),
    ("security", "P2", True, False, "Credible suspicious activity, not yet confirmed; security is always reviewed.",
     "Send to the security team to investigate before any compromise.",
     ["Several people got emails asking them to sign in to a fake {sys} page; nobody clicked yet.",
      "We see failed sign-in attempts on {name}'s {sys} account from an unfamiliar location, but no success.",
      "{name} received an unexpected sign-in approval request for the {sys} they did not start."]),
    ("general", "P4", False, False, "A general question with low impact.", "Reply with the requested information.",
     ["What are the support hours during the {month} holidays?",
      "Is there a training session on the {sys} next week?",
      "Who should {dept} contact for general IT questions?"]),
    ("general", "UNKNOWN", False, True, "The request is too vague to judge impact.",
     "Ask what the issue is and who is affected.",
     ["Can someone take a look at our setup?", "We have a question, please call me back.",
      "Not sure who to ask about this, can you help?"]),
]

WORD = re.compile(r"[a-z0-9]+")


def words(t: str) -> set[str]:
    return set(WORD.findall(t.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def fill(t: str, rng: random.Random) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: rng.choice(S[m.group(1)]), t)


def examples(ticket: str, g) -> list[dict]:
    cat, pri, hr, nmi, reason, action, _ = g
    call = {"id": "call_1", "type": "function",
            "function": {"name": "lookup_triage_policy", "arguments": {"category": cat}}}
    final = json.dumps({"category": cat, "priority": pri, "human_review": hr, "needs_more_info": nmi,
                        "reason": reason, "next_action": action})
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}]
    a = {"messages": base + [{"role": "assistant", "tool_calls": [call]}], "tools": TOOLS}
    b = {"messages": base + [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "name": "lookup_triage_policy",
         "content": json.dumps(lookup_triage_policy(cat))},
        {"role": "assistant", "content": final}], "tools": TOOLS}
    return [a, b]


def main() -> None:
    rng = random.Random(SEED)
    evals = [json.loads(l)["prompt"] for l in Path("evals/triage_cases.jsonl").read_text().splitlines() if l.strip()]
    eval_words = [words(e) for e in evals]
    tickets, rejected_sim, rejected_guard, max_sim = [], 0, 0, 0.0
    for g in GROUPS:
        seen, tries = set(), 0
        while len(seen) < TARGET_PER_GROUP and tries < 400:
            tries += 1
            t = fill(rng.choice(g[6]), rng)
            if t in seen:
                continue
            if screen_text(t) is not None:
                rejected_guard += 1
                continue
            sim = max(jaccard(words(t), e) for e in eval_words)
            if sim >= SIM_THRESHOLD:
                rejected_sim += 1
                continue
            max_sim = max(max_sim, sim)
            seen.add(t)
            tickets.append((t, g))
    rng.shuffle(tickets)
    n_valid = max(1, len(tickets) // 10)
    out = Path("finetune/data")
    out.mkdir(parents=True, exist_ok=True)
    for name, chunk in (("valid", tickets[:n_valid]), ("train", tickets[n_valid:])):
        with (out / f"{name}.jsonl").open("w") as f:
            for t, g in chunk:
                for ex in examples(t, g):
                    f.write(json.dumps(ex) + "\n")
    by = {}
    for _, g in tickets:
        k = f"{g[0]}/{g[1]}"
        by[k] = by.get(k, 0) + 1
    stats = {"tickets": len(tickets), "train_tickets": len(tickets) - n_valid, "valid_tickets": n_valid,
             "examples": 2 * len(tickets), "rejected_too_similar_to_eval": rejected_sim,
             "rejected_by_guardrail": rejected_guard, "max_similarity_to_eval": round(max_sim, 3),
             "by_category_priority": dict(sorted(by.items()))}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
