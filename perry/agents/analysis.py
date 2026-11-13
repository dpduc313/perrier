"""Analysis Agent: answers questions about Datasets with read-only SQL, and with Python
run in the Sandbox when SQL isn't enough.

Interface: question in, findings out, always followed by the SQL and Python that
produced them, so the user can check the logic. Charts travel as chart markers.
"""

import csv
import difflib
import io
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage
from langchain_core.tools import ToolException, tool

from perry import charts, datasets, models, retries, sandbox, usage

SYSTEM_PROMPT = """You are Perry's Analysis Agent. You answer questions about the user's \
Datasets (tables uploaded from CSV or spreadsheet files) by querying them with SQL.

- Call list_datasets first unless you already know the Dataset and its columns.
- Before writing SQL, call describe_datasets for the Datasets you will query, all in one \
call: it gives each column's type and the values it holds, so filter on values that \
actually occur.
- Write one PostgreSQL SELECT statement per run_sql call. Refer to Datasets by their name, \
which is the table name. Let SQL do the computing (sums, averages, grouping, filtering).
- Your access is read-only. If a query fails, read the error, fix the query and try again \
(at most 3 attempts). If it still fails, say so plainly with the error.
- Results are cut at {row_limit} rows; aggregate rather than listing many rows.
- When SQL isn't enough (statistics, forecasts, reshaping, charts), call run_python. \
Give it `inputs`: CSV file names mapped to SELECT statements; each result is saved as \
that file in the script's working directory, e.g. inputs {{"sales.csv": "SELECT ..."}} \
for pd.read_csv("sales.csv"). The script has pandas, numpy, scipy and \
matplotlib, but no network and no other files. Print the results you need and save each \
chart with plt.savefig("<name>.png"); charts are shown to the user automatically.
- Answer only from query and script results. Never invent numbers.
- Be concise: report the figures, not a polished essay."""

# Python inputs can be larger than the rows the model reads back from run_sql.
INPUT_ROW_LIMIT = int(os.getenv("PERRY_SANDBOX_INPUT_ROW_LIMIT", "100000"))


@tool
def list_datasets() -> str:
    """List the user's Datasets with what one row is, their columns and types, and their
    join hints."""
    found = datasets.list_datasets()
    if not found:
        return "The user has not uploaded any Datasets yet."
    return "\n".join(d.describe() for d in found)


@tool
def describe_datasets(names: list[str]) -> str:
    """Describe the named Datasets column by column: each column's type, what it means
    when the user said so, its share of empty values, its min and max, its distinct values
    or a few examples, and the Dataset column it references, if any: declared by the user
    or inferred, with how many of its values are not found there (a plain JOIN drops
    those rows)."""
    found = {d.name: d for d in datasets.list_datasets()}
    parts = []
    for name in names:
        dataset = found.get(name) or found.get(datasets.identifier(name, "dataset"))
        parts.append(dataset.describe_columns() if dataset else _unknown(name, list(found)))
    return "\n\n".join(parts) or "Name the Datasets to describe."


def _unknown(name: str, names: list[str]) -> str:
    if not names:
        return f"No Dataset is called {name!r}: the user has not uploaded any Datasets yet."
    typed = datasets.identifier(name, "dataset")
    part_of = [n for n in names if typed in n or n in typed]
    alike = difflib.get_close_matches(typed, names, n=3, cutoff=0.6)
    close = list(dict.fromkeys([*alike, *part_of]))[:3]
    if close:
        return f"No Dataset is called {name!r}. Did you mean {' or '.join(close)}?"
    return f"No Dataset is called {name!r}. The Datasets are: {', '.join(names)}."


@tool(response_format="content")
def run_sql(query: str) -> str:
    """Run one read-only SQL SELECT statement over the Datasets and return the rows."""
    try:
        result = datasets.run_query(query)
    except datasets.QueryError as error:
        raise ToolException(str(error)) from error
    return format_table(result)


run_sql.handle_tool_error = True  # the Agent sees the error and can fix its query


@tool(response_format="content")
def run_python(code: str, inputs: dict[str, str] | None = None) -> str:
    """Run a Python script in an isolated sandbox. `inputs` maps CSV file names to
    read-only SQL SELECT statements whose results the script can read, e.g.
    {"sales.csv": "SELECT ..."} for pd.read_csv("sales.csv"). Returns what the script
    printed, any CSV tables it saved, and its charts (PNG files it saved)."""
    files, notes = {}, []
    for name, query in (inputs or {}).items():
        # Models often name an input without ".csv", then read "<name>.csv".
        name = name if Path(name).suffix else f"{name}.csv"
        try:
            result = datasets.run_query(query, row_limit=INPUT_ROW_LIMIT)
        except datasets.QueryError as error:
            raise ToolException(f"Input {name}: {error}") from error
        files[name] = _csv(result)
        if result.truncated:
            notes.append(f"{name} was cut at {INPUT_ROW_LIMIT} rows.")
    try:
        outcome = sandbox.run(code, files)
    except sandbox.SandboxUnavailable as error:
        raise ToolException(f"Python can't run right now: {error}") from error
    if not outcome.ok:
        printed = f"\nIt printed before failing:\n{outcome.output}" if outcome.output else ""
        given = f"Its input files: {', '.join(files)}." if files else "It had no input files."
        raise ToolException(f"{outcome.error}{printed}\n{given}")
    parts = [*notes, outcome.output.strip() or "(the script printed nothing)"]
    parts += [f"{name}:\n{table.strip()}" for name, table in outcome.tables.items()]
    if outcome.charts:
        parts.append("Charts (shown to the user): " + " ".join(map(charts.save, outcome.charts)))
    return "\n\n".join(parts)


run_python.handle_tool_error = True  # the Agent sees the error and can fix its script


def _csv(result: datasets.QueryResult) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(result.columns)
    writer.writerows(result.rows)
    return out.getvalue().encode()


def format_table(result: datasets.QueryResult) -> str:
    if not result.columns:
        return "The query returned no columns."
    lines = [" | ".join(result.columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in result.rows]
    if not result.rows:
        lines.append("(no rows)")
    if result.truncated:
        lines.append(f"(only the first {len(result.rows)} rows are shown)")
    return "\n".join(lines)


def analysis_agent():
    # Built per call (cheap) so it always uses the current model provider.
    return create_agent(
        models.model_for("analysis"),
        tools=[list_datasets, describe_datasets, run_sql, run_python],
        middleware=[retries.RetryFailedSteps()],
        system_prompt=SYSTEM_PROMPT.format(row_limit=datasets.ROW_LIMIT),
        name="analysis_agent",
    )


def ask(question: str) -> str:
    messages: list[BaseMessage] = []
    try:
        # Streamed, so that if the Turn's token ceiling stops the Agent before it answers,
        # the results of its queries and scripts still reach the user.
        for state in analysis_agent().stream(
            {"messages": [{"role": "user", "content": question}]}, stream_mode="values"
        ):
            messages = state["messages"]
    except usage.TurnTokenCeilingReached as ceiling:
        if found := _results_so_far(messages):
            ceiling.findings.append(_with_the_logic(found, messages))
        raise
    return _with_the_logic(messages[-1].text, messages)


def _results_so_far(messages: list[BaseMessage]) -> str:
    return "\n\n".join(
        m.text
        for m in messages
        if m.type == "tool" and m.name in ("run_sql", "run_python") and m.status != "error"
    )


def _with_the_logic(findings: str, messages: list[BaseMessage]) -> str:
    """`findings` followed by the SQL and Python that produced them."""
    # Code, not the model, keeps every chart made in this run in the findings.
    findings = charts.keep(findings, [m.text for m in messages if m.type == "tool"])
    shown = [findings]
    queries = [args["query"] for args in calls_that_ran(messages, "run_sql")]
    if queries:
        label = "Query used" if len(queries) == 1 else "Queries used"
        shown.append(f"{label}:\n" + "\n\n".join(f"```sql\n{q}\n```" for q in queries))
    for args in calls_that_ran(messages, "run_python"):
        reads = "".join(f"\n- {name}: `{q}`" for name, q in (args.get("inputs") or {}).items())
        reading = f", reading:{reads}" if reads else ""
        shown.append(f"Python used{reading}\n```python\n{args['code']}\n```")
    return "\n\n".join(shown)


def calls_that_ran(messages: list[BaseMessage], tool_name: str) -> list[dict]:
    """The arguments of every successful call to the tool, in order."""
    succeeded = {m.tool_call_id for m in messages if m.type == "tool" and m.status != "error"}
    return [
        call["args"]
        for m in messages
        if m.type == "ai"
        for call in m.tool_calls
        if call["name"] == tool_name and call["id"] in succeeded
    ]
