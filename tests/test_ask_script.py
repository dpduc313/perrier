"""`python -m scripts.ask`: live questions that record each Turn's time and cost."""

import json

from fakes import call, last_message, with_usage

from perry import threads
from scripts import ask


def add_budget(perry) -> None:
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")


def test_a_turn_is_recorded_with_its_found_with_line_and_every_model_call(perry, owner, prices):
    prices({"scripted": {"input": 1.0, "output": 2.0}})
    add_budget(perry)
    perry.script(
        with_usage(call("ask_knowledge_agent", question="Q3 marketing budget?"), 1000, 100),
        call("hybrid_search", question="Q3 marketing budget"),
        last_message,  # Knowledge Agent
        last_message,  # Supervisor
    )

    row = ask.ask(owner, "How big is my Q3 marketing budget?", knowledge_only=False, technique=None)

    assert row["reply"].startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert row["found_with"] == "Hybrid search (picked logically)"
    assert row["mode"] == "turn"
    chat = next(c for c in row["calls"] if c["model"] == "scripted" and c["kind"] == "chat")
    assert (chat["calls"], chat["input_tokens"], chat["output_tokens"]) == (4, 1000, 100)
    assert row["cost_usd"] == (1000 * 1.0 + 100 * 2.0) / 1_000_000
    assert threads.list_recent(owner) == []  # the Thread made for it is deleted
    stages = row["stages"]
    # The Knowledge Agent's own two model calls are inside its stage, not the Supervisor's.
    assert (stages["choosing_calls"], stages["answering_calls"]) == (1, 1)
    assert [agent["name"] for agent in stages["agents"]] == ["ask_knowledge_agent"]
    assert stages["agents_s"] <= row["seconds"]


def test_the_knowledge_agent_can_be_asked_alone(perry, owner):
    add_budget(perry)
    perry.script(call("hybrid_search", question="Q3 marketing budget"), last_message)

    row = ask.ask(owner, "Q3 marketing budget?", knowledge_only=True, technique=None)

    assert row["mode"] == "knowledge"
    assert row["found_with"] == "Hybrid search (picked logically)"
    assert sum(c["calls"] for c in row["calls"] if c["kind"] == "chat") == 2


def test_every_turn_is_appended_to_the_log_with_its_settings(
    perry, owner, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(ask, "LOG", tmp_path / "asks.jsonl")
    perry.script("Hello!", "Hello again!")

    assert ask.main(["Hi there", "--repeat", "2", "--label", "try"]) == 0

    rows = [json.loads(line) for line in (tmp_path / "asks.jsonl").read_text().splitlines()]
    assert [row["reply"] for row in rows] == ["Hello!", "Hello again!"]
    assert {row["label"] for row in rows} == {"try"}
    assert "llm_model" in rows[0]["settings"]
    assert rows[0]["stages"]["agents"] == []  # small talk asks no Agent
    out = capsys.readouterr().out
    assert "2 Turns: median" in out
    assert "Supervisor writes answer" in out
