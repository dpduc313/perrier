"""Ask Perry one or more questions from the command line, against the real models, and
record what each Turn cost: time, the "Found with" line, and every model call it made.

Run from the repo root (uses DATABASE_URL from .env, so your own Documents):

    uv run python -m scripts.ask "How is the Q3 marketing budget split?"
    uv run python -m scripts.ask "Q1" "Q2" --repeat 3        # each question 3 times
    uv run python -m scripts.ask "Q1" --knowledge            # the Knowledge Agent only
    uv run python -m scripts.ask "Q1" --technique hyde       # force a Retrieval Technique
    uv run python -m scripts.ask "Q1" --label "inet-qwen down"

A full Turn goes through the Supervisor as the Owner, on a Thread made for it and deleted
afterwards (its Usage Ledger rows are kept), and its time is split into stages: the
Supervisor choosing Agents, the Agents running, and the Supervisor writing the answer.
Every Turn is appended to
scripts/results/asks.jsonl with the settings it ran under and the git commit, so runs on
two branches, or with two settings, can be compared. Paid model calls cost real money and
count toward today's Daily Cap.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from langchain_core.tracers.context import collect_runs

from perry import db, models, threads, usage, users  # importing perry loads .env

LOG = Path(__file__).parent / "results" / "asks.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ask", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("questions", nargs="+", help="what to ask, one Turn each")
    parser.add_argument("--repeat", type=int, default=1, help="ask each question this many times")
    parser.add_argument("--knowledge", action="store_true", help="ask the Knowledge Agent only")
    parser.add_argument("--technique", help="force this Retrieval Technique, e.g. hybrid_search")
    parser.add_argument("--label", default="", help="a note saved with each row, e.g. the setup")
    parser.add_argument("--no-log", action="store_true", help=f"don't append to {LOG.name}")
    args = parser.parse_args(argv)

    owner = users.require_owner()
    rows = []
    for question in args.questions:
        for _ in range(args.repeat):
            row = ask(owner, question, args.knowledge, args.technique)
            row["label"] = args.label
            print_row(row)
            rows.append(row)
            if not args.no_log:
                LOG.parent.mkdir(exist_ok=True)
                with LOG.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
    if len(rows) > 1:
        print_summary(rows)
    return 0


def ask(owner, question: str, knowledge_only: bool, technique: str | None) -> dict:
    """One Turn: the reply, how long it took and what it cost, as one log row."""
    turn_id = usage.new_turn_id()
    started = time.perf_counter()
    if knowledge_only:
        from perry.agents import knowledge

        with usage.turn(owner, None, turn_id):
            findings = knowledge.find(question, technique)
        reply = knowledge.with_found_with(findings.text, findings.found_with)
    else:
        from perry import supervisor

        thread_id = threads.create(owner, question)
        try:
            with collect_runs() as runs:
                reply = supervisor.run_turn(
                    owner, thread_id, question, turn_id=turn_id, technique=technique
                )
        finally:
            threads.delete(owner, thread_id)
        if not isinstance(reply, str):  # a Confirmation: nothing was performed
            reply = f"(waiting for a Confirmation, not performed: {reply})"
    seconds = time.perf_counter() - started
    calls = turn_calls(turn_id)
    stages = None if knowledge_only else turn_stages(runs.traced_runs, seconds)
    return {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "settings": settings(),
        "mode": "knowledge" if knowledge_only else "turn",
        "technique": technique,
        "question": question,
        "seconds": round(seconds, 1),
        "stages": stages,
        "found_with": next(
            (
                line.removeprefix("Found with: ")
                for line in reply.splitlines()
                if line.startswith("Found with: ")
            ),
            None,
        ),
        "cost_usd": round(sum(call["cost_usd"] for call in calls), 6),
        "tokens": sum(call["input_tokens"] + call["output_tokens"] for call in calls),
        "calls": calls,
        "reply": reply,
    }


def turn_calls(turn_id: str) -> list[dict]:
    """The Turn's model calls from the Usage Ledger, grouped by model and kind."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            """SELECT model, kind, count(*) AS calls, sum(input_tokens) AS input_tokens,
                      sum(output_tokens) AS output_tokens, sum(cost_usd) AS cost_usd
               FROM usage_ledger WHERE turn_id = %s
               GROUP BY model, kind ORDER BY min(id)""",
            (turn_id,),
        ).fetchall()
    return [
        {
            **row,
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "cost_usd": float(row["cost_usd"]),
        }
        for row in rows
    ]


def turn_stages(runs: list, seconds: float) -> dict:
    """How the Turn's `seconds` split, from its traced runs: the Supervisor's model calls
    up to its last Agent (choosing Agents), the Agents it asked (a tool call each, timed
    from the first one's start to the last one's end, as they may run at once), its model
    calls after that (writing the answer), and the rest (the Thread, checkpoints)."""
    from perry import supervisor

    # The traces don't nest (LangGraph's model node isn't traced), so an Agent is a call
    # of a Supervisor tool, and a Supervisor model call is one made while no Agent ran.
    agent_tools = {
        t.name
        for t in (
            supervisor.ask_knowledge_agent,
            supervisor.ask_task_agent,
            supervisor.ask_analysis_agent,
        )
    }
    flat, pending = [], list(runs)
    while pending:
        run = pending.pop()
        flat.append(run)
        pending.extend(run.child_runs or [])
    agents = sorted(
        (r for r in flat if r.run_type == "tool" and r.name in agent_tools),
        key=lambda agent: agent.start_time,
    )
    model_calls = []
    # A model chain's call wraps its model's, so only the outermost one counts.
    for r in sorted((r for r in flat if r.run_type == "llm"), key=lambda r: r.start_time):
        if any(a.start_time <= r.start_time <= a.end_time for a in agents):
            continue
        if not model_calls or r.start_time > model_calls[-1].end_time:
            model_calls.append(r)
    last_agent = agents[-1].start_time if agents else None
    choosing = [c for c in model_calls if last_agent and c.start_time < last_agent]
    answering = [c for c in model_calls if c not in choosing]
    agents_s = (
        (max(a.end_time for a in agents) - agents[0].start_time).total_seconds() if agents else 0.0
    )
    choosing_s = sum(_seconds(call) for call in choosing)
    answering_s = sum(_seconds(call) for call in answering)
    return {
        "choosing_s": round(choosing_s, 1),
        "choosing_calls": len(choosing),
        "agents_s": round(agents_s, 1),
        "agents": [{"name": a.name, "seconds": round(_seconds(a), 1)} for a in agents],
        "answering_s": round(answering_s, 1),
        "answering_calls": len(answering),
        "other_s": round(seconds - choosing_s - agents_s - answering_s, 1),
    }


def _seconds(run) -> float:
    return (run.end_time - run.start_time).total_seconds()


def settings() -> dict:
    """The model settings a row ran under."""
    return {
        "llm_model": models.LLM_MODEL,
        "fallback_llm_model": models.FALLBACK_LLM_MODEL,
        "inet_qwen": models.INET_QWEN_BASE_URL or None,
        "paid_chat_only": models.PAID_CHAT_ONLY,
        "chat_timeout": models.CHAT_TIMEOUT,
    }


def print_row(row: dict) -> None:
    print(f"\n=== {row['question']}  ({row['mode']}, {row['seconds']}s, ${row['cost_usd']:.4f})")
    print(row["reply"])
    print(f"\n  {'model':32} {'kind':14} {'calls':>5} {'in':>7} {'out':>6} {'cost':>9}")
    for call in row["calls"]:
        print(
            f"  {call['model'][:32]:32} {call['kind'][:14]:14} {call['calls']:5} "
            f"{call['input_tokens']:7} {call['output_tokens']:6} ${call['cost_usd']:8.4f}"
        )
    if row["stages"]:
        print_stages(row["stages"], row["seconds"])


def print_stages(stages: dict, seconds: float) -> None:
    agents = ", ".join(f"{a['name']} {a['seconds']}s" for a in stages["agents"])
    lines = [
        ("Supervisor chooses Agents", stages["choosing_s"], f"{stages['choosing_calls']} calls"),
        ("Agents run", stages["agents_s"], f"{len(stages['agents'])} asked: {agents or '-'}"),
        ("Supervisor writes answer", stages["answering_s"], f"{stages['answering_calls']} calls"),
        ("Other (Thread, checkpoints)", stages["other_s"], ""),
        ("Turn", seconds, ""),
    ]
    print(f"\n  {'stage':28} {'seconds':>8} {'share':>6}  detail")
    for name, stage_s, detail in lines:
        print(f"  {name:28} {stage_s:8.1f} {stage_s / seconds if seconds else 0:6.0%}  {detail}")


def print_summary(rows: list[dict]) -> None:
    seconds = sorted(row["seconds"] for row in rows)
    cost = sum(row["cost_usd"] for row in rows)
    print(
        f"\n{len(rows)} Turns: median {seconds[len(seconds) // 2]}s, "
        f"total ${cost:.4f}, ${cost / len(rows):.4f} per Turn"
    )
    turns = [row for row in rows if row["stages"]]
    if not turns:
        return
    print(f"\n  {'question':40} {'choose':>7} {'agents':>7} {'asked':>5} {'answer':>7} {'turn':>7}")
    for row in turns:
        stages = row["stages"]
        print(
            f"  {row['question'][:40]:40} {stages['choosing_s']:7.1f} {stages['agents_s']:7.1f} "
            f"{len(stages['agents']):5} {stages['answering_s']:7.1f} {row['seconds']:7.1f}"
        )
    print(
        f"  {'median':40} {_median(r['stages']['choosing_s'] for r in turns):7.1f} "
        f"{_median(r['stages']['agents_s'] for r in turns):7.1f} "
        f"{_median(len(r['stages']['agents']) for r in turns):5} "
        f"{_median(r['stages']['answering_s'] for r in turns):7.1f} "
        f"{_median(r['seconds'] for r in turns):7.1f}"
    )


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
