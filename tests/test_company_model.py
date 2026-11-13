"""Knowledge work on the company's model, inet-qwen (docs/adr/0006): the Knowledge Agent's
steps and its techniques' calls (or, with "techniques", only those calls) go there first,
as JSON choices since its server refuses tool calls, and to the chat models for any call
it can't answer."""

import json
import uuid
from dataclasses import replace

import httpx
import openai
import pytest
from conftest import PNG, Perry
from fakes import ScriptedChatModel, call, fake_provider, last_message
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from perry import db, models, usage
from perry.supervisor import Approve, Confirmation


def choose(action: str, answer: str = "", **arguments) -> AIMessage:
    """A JSON choice from the company's model, named as its server names it: a tool with
    its arguments, or "answer"."""
    step = (
        {"action": action, "answer": answer}
        if action == "answer"
        else {"action": action, "arguments": arguments}
    )
    return AIMessage(json.dumps({"step": step}), response_metadata={"model_name": "knowledge"})


def answer_with_results(messages) -> AIMessage:
    """The company's model answering with the latest search results, as a JSON choice."""
    results = messages[-1].text.split("\n", 1)[1]
    return choose("answer", answer=results)


def from_gemini(step) -> AIMessage:
    """A reply from the chat model, named as Gemini names its replies."""
    if isinstance(step, str):
        message = AIMessage(step)
    else:
        tool_call = {"name": step.name, "args": step.args, "id": f"call_{uuid.uuid4().hex}"}
        message = AIMessage("", tool_calls=[tool_call])
    message.response_metadata = {"model_name": "gemini-3.8-flash"}
    return message


def unreachable() -> Exception:
    return openai.APIConnectionError(request=httpx.Request("POST", "https://inet.example/v1"))


def server_error() -> Exception:
    request = httpx.Request("POST", "https://inet.example/v1")
    return openai.InternalServerError(
        "Internal Server Error", response=httpx.Response(500, request=request), body=None
    )


def timed_out() -> Exception:
    return openai.APITimeoutError(request=httpx.Request("POST", "https://inet.example/v1"))


@pytest.fixture
def company() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def perry(owner, chat_model, company, tmp_path):
    """A Perry whose knowledge work goes to `company` first and everything else (and any
    knowledge-work call `company` can't answer) to `chat_model`."""
    provider = replace(fake_provider(chat_model), company_chat=lambda: company)
    with models.use(provider):
        yield Perry(chat_model, tmp_path, owner)


def ask_about_budget(perry, **send) -> str:
    perry.add_document("budget.md", "The marketing budget for Q3 is 40,000 dollars.")
    perry.add_document("hiking.md", "Trail notes: the ridge path is steep after the waterfall.")
    return perry.send("How big is my Q3 marketing budget?", **send)


def ledger() -> list[tuple[str, str]]:
    with db.pool().connection() as conn:
        rows = conn.execute("SELECT kind, model FROM usage_ledger ORDER BY id").fetchall()
    return [(row["kind"], row["model"]) for row in rows]


def test_the_knowledge_agent_works_on_the_company_model_and_the_supervisor_does_not(perry, company):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        last_message,  # Supervisor passes the findings on
    )
    company.steps = [choose("hybrid_search", question="Q3 marketing budget"), answer_with_results]

    reply = ask_about_budget(perry)

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    *_, citations, found_with = reply.splitlines()
    assert citations.startswith("Citations: [budget.md]")
    assert found_with == "Found with: Hybrid search (picked logically) · inet-qwen"
    assert "hybrid_search" not in perry.model.bound_tools  # only the Supervisor bound tools


def test_the_company_model_reads_tools_and_their_results_as_plain_messages(perry, company):
    perry.script(call("ask_knowledge_agent", question="Q3 marketing budget?"), last_message)
    company.steps = [choose("hybrid_search", question="Q3 marketing budget"), answer_with_results]

    ask_about_budget(perry)

    first, second = company.calls
    system = first[0]
    assert isinstance(system, SystemMessage)
    assert "- hybrid_search(question: string): " in system.text  # every offered technique
    assert not any(isinstance(m, ToolMessage) for m in second)
    assert json.loads(second[-2].text) == {
        "step": {"action": "hybrid_search", "arguments": {"question": "Q3 marketing budget"}}
    }
    assert isinstance(second[-1], HumanMessage)
    assert second[-1].text.startswith("Result of hybrid_search:\n[budget.md]")


def test_a_choice_of_a_technique_not_on_offer_is_answered_by_the_chat_model(perry, company):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        from_gemini(call("hybrid_search", question="Q3 marketing budget")),  # instead of hyde
        last_message,  # Supervisor
    )
    # Only Hybrid search is offered (forced), so choosing HyDE isn't allowed.
    company.steps = [choose("hyde", question="Q3 marketing budget"), answer_with_results]

    reply = ask_about_budget(perry, technique="hybrid_search")

    assert reply.splitlines()[-1] == "Found with: Hybrid search (forced) · inet-qwen"
    assert len(company.calls) == 2


@pytest.mark.parametrize("failure", [unreachable, server_error, timed_out])
def test_a_step_the_company_model_cant_answer_goes_to_the_chat_model_and_the_next_step_tries_it_again(
    perry, company, failure
):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        from_gemini(call("hybrid_search", question="Q3 marketing budget")),
        last_message,  # Supervisor
    )
    company.steps = [failure(), answer_with_results]

    reply = ask_about_budget(perry)

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert reply.splitlines()[-1] == "Found with: Hybrid search (picked logically) · inet-qwen"
    assert len(company.calls) == 2


def test_the_found_with_line_names_the_chat_model_when_it_gave_the_answer(perry, company):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        from_gemini("The marketing budget for Q3 is 40,000 dollars. [budget.md]"),
        last_message,  # Supervisor
    )
    company.steps = [choose("hybrid_search", question="Q3 marketing budget"), unreachable()]

    reply = ask_about_budget(perry)

    assert reply.splitlines()[-1] == (
        "Found with: Hybrid search (picked logically) · Gemini (inet-qwen unavailable)"
    )


def test_a_bug_is_not_hidden_by_the_chat_model(perry, company):
    perry.script(call("ask_knowledge_agent", question="What is the Q3 marketing budget?"))
    company.steps = [ValueError("bad request")]

    with pytest.raises(ValueError, match="bad request"):
        ask_about_budget(perry)


def test_techniques_helper_calls_go_to_the_company_model_at_no_cost(perry, company, caplog):
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),
        last_message,  # Supervisor
    )
    company.steps = [
        choose("hyde", question="Q3 marketing budget"),
        AIMessage(  # HyDE's hypothetical passage: a plain call, no JSON choice
            "The marketing budget for Q3 is set at 40,000 dollars.",
            response_metadata={"model_name": "knowledge"},
        ),
        answer_with_results,
    ]

    reply = ask_about_budget(perry, technique="hyde")

    assert reply.splitlines()[-1] == "Found with: HyDE (forced) · inet-qwen"
    assert ("hyde", "knowledge") in ledger()
    assert [kind for kind, model in ledger() if model == "knowledge"] == ["chat", "hyde", "chat"]
    assert usage.cost("knowledge", 1000, 1000) == 0  # priced at zero, not just unpriced
    assert "'knowledge'" not in caplog.text


def test_scanned_pages_are_read_by_the_company_model(monkeypatch, company):
    company.steps = [AIMessage("INVOICE 4471", response_metadata={"model_name": "knowledge"})]
    gemini = ScriptedChatModel()
    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "https://inet.example/v1")
    monkeypatch.setattr(models, "_inet_qwen_chat", lambda: company)

    read = models._read_scanned_page(PNG, "image/png", lambda: gemini)

    assert (read.text, read.model) == ("INVOICE 4471", "knowledge")
    assert gemini.calls == []


def test_scanned_pages_go_to_gemini_when_the_company_model_is_unreachable(monkeypatch, company):
    company.steps = [unreachable()]
    gemini = ScriptedChatModel(steps=["INVOICE 4471"])
    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "https://inet.example/v1")
    monkeypatch.setattr(models, "_inet_qwen_chat", lambda: company)

    read = models._read_scanned_page(PNG, "image/png", lambda: gemini)

    assert read.text == "INVOICE 4471"
    assert read.model == models.LLM_MODEL


def test_without_a_company_model_knowledge_work_runs_on_the_chat_model(owner, chat_model):
    with models.use(fake_provider(chat_model)):
        assert models.knowledge_model() is models.chat_model()


def test_one_setting_sends_chat_straight_to_the_paid_model(monkeypatch):
    from langchain_openai import ChatOpenAI

    monkeypatch.setattr(models, "PAID_CHAT_ONLY", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    chat = models._default.__wrapped__().chat_model()

    assert isinstance(chat, ChatOpenAI)
    assert chat.model_name == models.FALLBACK_LLM_MODEL


def test_by_default_chat_tries_the_free_model_first(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    chat = models._default.__wrapped__().chat_model()

    assert chat._llm_type == "model-chain"


def test_the_found_with_line_names_every_model_that_gave_way(owner, chat_model, company, tmp_path):
    free, paid = ScriptedChatModel(model_name="gemini-3.8-flash"), ScriptedChatModel()
    provider = replace(
        fake_provider(chat_model),
        chat_model=lambda: models.free_first(free, paid),
        company_chat=lambda: company,
    )
    paid_answer = AIMessage(
        "The marketing budget for Q3 is 40,000 dollars. [budget.md]",
        response_metadata={"model_name": "gpt-5.4-mini-2026-03-17"},
    )
    free.steps = [
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        models.RateLimitError("429 quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
        last_message,  # Supervisor
    ]
    company.steps = [choose("hybrid_search", question="Q3 marketing budget"), unreachable()]
    paid.steps = [paid_answer]

    with models.use(provider):
        reply = ask_about_budget(Perry(free, tmp_path, owner))

    assert reply.splitlines()[-1] == (
        "Found with: Hybrid search (picked logically) · OpenAI (inet-qwen, Gemini unavailable)"
    )


# The whole stage (docs/adr/0007): whatever PERRY_INET_QWEN_FOR names asks the company's
# model first, with any tool as a JSON choice.


@pytest.fixture
def company_for(monkeypatch):
    def run_on_company(*roles: str) -> None:
        monkeypatch.setattr(models, "INET_QWEN_FOR", frozenset(roles))

    return run_on_company


def test_with_the_supervisor_there_too_a_whole_turn_runs_on_the_company_model(
    perry, company, company_for
):
    company_for("knowledge", "supervisor")
    company.steps = [
        choose("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        choose("hybrid_search", question="Q3 marketing budget"),  # Knowledge Agent
        answer_with_results,  # Knowledge Agent
        answer_with_results,  # Supervisor passes the findings on
    ]

    reply = ask_about_budget(perry)

    assert reply.startswith("The marketing budget for Q3 is 40,000 dollars.")
    assert reply.splitlines()[-1] == "Found with: Hybrid search (picked logically) · inet-qwen"
    assert perry.model.calls == []  # Gemini and OpenAI never asked
    assert perry.consulted() == ["ask_knowledge_agent"]
    supervisor_prompt = company.calls[0][0].text
    assert "- ask_knowledge_agent(question: string): " in supervisor_prompt
    assert "config" not in supervisor_prompt.split("The tools")[1]  # Perry's own argument


def test_a_write_action_chosen_on_the_company_model_still_waits_for_a_confirmation(
    perry, company, company_for, workspace
):
    company_for("task")
    perry.script(call("ask_task_agent", request="Email Sam that I'm running 10 minutes late."))
    company.steps = [
        choose(
            "send_email",
            to="sam@example.com",
            subject="Running late",
            body="Running 10 minutes late, start without me.",
        )
    ]

    pending = perry.send("Tell Sam I'm running 10 minutes late")

    assert isinstance(pending, Confirmation)
    assert pending.fields == {
        "to": "sam@example.com",
        "subject": "Running late",
        "body": "Running 10 minutes late, start without me.",
    }
    assert workspace.sent == []

    company.steps = [lambda messages: choose("answer", answer=messages[-1].text)]  # Task Agent
    perry.script(last_message)  # Supervisor
    reply = perry.resume(Approve())

    assert [e.subject for e in workspace.sent] == ["Running late"]
    assert "Sent" in reply


def test_a_choice_missing_an_argument_is_answered_by_the_chat_model(
    perry, company, company_for, workspace
):
    company_for("task")
    perry.script(
        call("ask_task_agent", request="Email Sam that I'm running late."),  # Supervisor
        from_gemini(call("send_email", to="sam@example.com", subject="Late", body="Running late.")),
    )
    company.steps = [choose("send_email", to="sam@example.com")]  # no subject or body

    pending = perry.send("Tell Sam I'm running late")

    assert isinstance(pending, Confirmation)
    assert pending.fields["subject"] == "Late"


def test_the_analysis_agent_queries_datasets_from_the_company_model(
    perry, company, company_for, tmp_path
):
    from perry import datasets

    company_for("analysis")
    path = tmp_path / "sales.csv"
    path.write_text("region,amount\nNorth,100\nSouth,250\nNorth,50\n", encoding="utf-8")
    datasets.upload(path)
    sql = "SELECT region, sum(amount) AS total FROM sales GROUP BY region ORDER BY region"
    perry.script(call("ask_analysis_agent", question="Total sales per region?"), last_message)
    company.steps = [
        choose("run_sql", query=sql),
        lambda messages: choose("answer", answer=f"Totals: {messages[-1].text}"),
    ]

    reply = perry.send("What were total sales per region?")

    assert "North | 150" in reply and "South | 250" in reply
    assert "- run_python(code: string, inputs?: object): " in company.calls[0][0].text


def test_one_setting_names_what_runs_on_the_company_model():
    assert models._company_model_for("knowledge") == {"knowledge"}
    assert models._company_model_for(" Knowledge, supervisor ") == {"knowledge", "supervisor"}
    assert models._company_model_for("all") == set(models.COMPANY_MODEL_ROLES)
    assert models._company_model_for("") == set()
    with pytest.raises(ValueError, match="router"):
        models._company_model_for("knowledge,router")


def test_who_answers_names_the_models_and_what_asks_the_company_model_first(monkeypatch):
    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "http://inet-qwen.test/v1")
    monkeypatch.setattr(models, "INET_QWEN_FOR", frozenset({"knowledge", "analysis"}))
    monkeypatch.setattr(models, "paid_chat_available", lambda: True)

    assert models.who_answers() == {
        "chat_model": models.LLM_MODEL,
        "fallback_model": models.FALLBACK_LLM_MODEL,
        "inet_qwen": models.INET_QWEN_MODEL,
        "inet_qwen_for": "analysis,knowledge",
    }
    paid = models.who_answers(paid=True)
    assert (paid["chat_model"], paid["fallback_model"]) == (models.BENCHMARK_LLM_MODEL, None)


def test_who_answers_without_a_company_model_names_none(monkeypatch):
    monkeypatch.setattr(models, "INET_QWEN_BASE_URL", "")

    answers = models.who_answers()

    assert (answers["inet_qwen"], answers["inet_qwen_for"]) == (None, None)


def test_with_only_the_techniques_there_the_knowledge_agent_steps_on_the_chat_model(
    perry, company, company_for
):
    company_for("techniques")
    perry.script(
        call("ask_knowledge_agent", question="What is the Q3 marketing budget?"),  # Supervisor
        call("hyde", question="Q3 marketing budget"),  # Knowledge Agent
        last_message,  # Knowledge Agent reports the Chunks it found
        last_message,  # Supervisor
    )
    company.steps = [
        AIMessage(  # HyDE's hypothetical passage
            "The marketing budget for Q3 is set at 40,000 dollars.",
            response_metadata={"model_name": "knowledge"},
        )
    ]

    reply = ask_about_budget(perry, technique="hyde")

    assert "40,000 dollars" in reply
    assert [kind for kind, model in ledger() if model == "knowledge"] == ["hyde"]
    assert len(company.calls) == 1
