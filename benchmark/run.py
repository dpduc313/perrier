"""Run the knowledge benchmark (ADR-0001): Perry's Knowledge Agent vs. QA-bot (LightRAG).

    uv run python -m benchmark.run qabot           # ingest + answer with QA-bot
    uv run python -m benchmark.run perry           # ingest + answer with Perry
    uv run python -m benchmark.run report          # score both -> results/RESULTS.md
    uv run python -m benchmark.run perry-retrieval [--reader docling,pymupdf,vision]
                                    [--chunking heading,fixed] [--yes]
                                    # no chat model: page recall of hybrid search,
                                    # without and with reranking, for each Reader and
                                    # Chunking Strategy (vision: paid page reading)
    uv run python -m benchmark.run grid [--only SETUP,...] [--rerun] [--yes]
                                    # every Retrieval Technique forced and every picking
                                    # mode -> results/grid.jsonl and results/GRID.md
    uv run python -m benchmark.run grid-report     # redraw results/GRID.md, no model calls

Answers are appended to results/answers-<system>.jsonl as they arrive. A run that
stops (e.g. the daily quota runs out) resumes from the first unanswered question.

QA-bot runs in its own virtualenv through benchmark/qabot_adapter.py. Set QABOT_DIR
if it is not checked out next to Perry. Its graph is stored under
benchmark/.storage/, so the QA-bot checkout itself is never written to.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from benchmark.corpus import DOCUMENTS_DIR
from benchmark.scoring import QUESTION_TYPES, Question, load_questions, score

HERE = Path(__file__).parent
QUESTIONS_PATH = HERE / "questions.toml"
RESULTS_DIR = HERE / "results"
STORAGE_DIR = HERE / ".storage"
SYSTEMS = ("perry", "qabot")

# Perry gets its own database so benchmark Documents never mix with real ones.
BENCHMARK_DATABASE_URL = "postgresql://perry:perry@127.0.0.1:5433/perry_benchmark"
QABOT_DIR = Path(os.getenv("QABOT_DIR", HERE.parent.parent / "QA-bot"))
QABOT_MODE = os.getenv("QABOT_MODE", "mix")
# Pause between questions, on top of each system's own pacing.
PAUSE_SECONDS = float(os.getenv("BENCHMARK_PAUSE_SECONDS", "0"))


def answers_path(system: str) -> Path:
    return RESULTS_DIR / f"answers-{system}.jsonl"


def load_answers(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["id"]: row for row in rows}


def answer_all(questions: list[Question], ask: Callable[[str], str], path: Path) -> None:
    """Ask every not-yet-answered question, saving each answer as soon as it arrives.

    The first failure stops the run (usually an exhausted quota, where carrying on
    only burns time); the failed question is retried on the next run.
    """
    done = load_answers(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for question in questions:
        if question.id in done:
            continue
        started = time.monotonic()
        answer = ask(question.question)
        row = {
            "id": question.id,
            "question": question.question,
            "answer": answer,
            "seconds": round(time.monotonic() - started, 1),
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{question.id}: answered in {row['seconds']}s", flush=True)
        if PAUSE_SECONDS:
            time.sleep(PAUSE_SECONDS)


# --- Scoring and report ------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    questions: int
    answered: int
    # Means over answered questions.
    correct: float = 0.0
    fact_recall: float = 0.0
    citation_recall: float = 0.0
    citation_precision: float = 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarise(
    questions: list[Question], answers: dict[str, dict[str, dict]]
) -> dict[tuple[str, str], Summary]:
    """(question type, system) -> scores. "all" is the type covering every question."""
    summary = {}
    for qtype in (*QUESTION_TYPES, "all"):
        group = [q for q in questions if qtype in ("all", q.type)]
        if not group:
            continue
        for system, by_id in answers.items():
            scores = [score(q, by_id[q.id]["answer"]) for q in group if q.id in by_id]
            if not scores:
                summary[(qtype, system)] = Summary(len(group), 0)
                continue
            summary[(qtype, system)] = Summary(
                questions=len(group),
                answered=len(scores),
                correct=_mean([float(s.correct) for s in scores]),
                fact_recall=_mean([s.fact_recall for s in scores]),
                citation_recall=_mean([s.citation_recall for s in scores]),
                citation_precision=_mean([s.citation_precision for s in scores]),
            )
    return summary


def _pct(value: float) -> str:
    return f"{value:.0%}"


def render_report(questions: list[Question], answers: dict[str, dict[str, dict]]) -> str:
    lines = [
        "# Knowledge benchmark results",
        "",
        (
            "Generated by `uv run python -m benchmark.run report`. Scores are means over "
            "answered questions. *Correct*: every expected fact present. *Citation recall*: "
            "share of required source pages cited. *Citation precision*: share of cited "
            "pages that are required or related source pages."
        ),
        "",
        "## Per question type",
        "",
        (
            "| Type | System | Answered | Correct | Fact recall | Citation recall "
            "| Citation precision |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for (qtype, system), s in summarise(questions, answers).items():
        if not s.answered:
            lines.append(f"| {qtype} | {system} | 0/{s.questions} | - | - | - | - |")
            continue
        lines.append(
            f"| {qtype} | {system} | {s.answered}/{s.questions} | {_pct(s.correct)} "
            f"| {_pct(s.fact_recall)} | {_pct(s.citation_recall)} "
            f"| {_pct(s.citation_precision)} |"
        )

    lines += ["", "## Per question", ""]
    for q in questions:
        lines += [f"### {q.id} ({q.type}): {q.question}", ""]
        lines.append(f"Required pages: {', '.join(q.pages)}")
        lines.append("")
        for system, by_id in answers.items():
            if q.id not in by_id:
                lines += [f"**{system}**: not answered", ""]
                continue
            s = score(q, by_id[q.id]["answer"])
            lines.append(
                f"**{system}**: facts {_pct(s.fact_recall)}, citation recall "
                f"{_pct(s.citation_recall)}, precision {_pct(s.citation_precision)}, "
                f"cited {', '.join(s.cited) or 'nothing'}"
            )
            lines.append("")
            answer = by_id[q.id]["answer"].strip().replace("\n", "\n> ")
            lines += [f"> {answer}", ""]
    return "\n".join(lines)


# --- Systems -------------------------------------------------------------------------


def _use_benchmark_database() -> None:
    """Point Perry at perry_benchmark (created on first use). Call before importing perry."""
    import psycopg

    os.environ["DATABASE_URL"] = BENCHMARK_DATABASE_URL
    admin_url = BENCHMARK_DATABASE_URL.rsplit("/", 1)[0] + "/perry"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = 'perry_benchmark'"
        ).fetchone()
        if not exists:
            conn.execute("CREATE DATABASE perry_benchmark")


@dataclass(frozen=True)
class Ingested:
    """What ingesting the benchmark Documents with one Reader and Chunking Strategy took."""

    documents: int
    reread: int  # Documents read and chunked in this run, not already stored that way
    chunks: int
    seconds: float
    reading_cost_usd: float  # vision model calls that read pages


def ingest_perry(chunking: str | None = None, reader: str | None = None) -> Ingested:
    """Ingest the benchmark Documents with `chunking` and `reader` (PERRY_CHUNKING and
    PERRY_DOCUMENT_READER if None). Documents stored another way are read again."""
    _use_benchmark_database()
    from perry import db, documents
    from perry.ingest import ensure_indexes_current, ingest_file

    db.init_schema()
    ensure_indexes_current()  # e.g. Documents stored before ColBERT or a model change
    paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if (reader or documents.READER) == "docling":
        documents.parse(paths[0], chunking, "docling")  # load Docling's models, untimed
    documents.forget_parsed()  # time reading, not a previous run's kept result
    with db.pool().connection() as conn:
        since = conn.execute("SELECT now() AS now").fetchone()["now"]
    started = time.monotonic()
    results = []
    for path in paths:
        results.append(ingest_file(path, chunking=chunking, reader=reader))
        print(results[-1].describe(), flush=True)
    seconds = time.monotonic() - started
    with db.pool().connection() as conn:
        chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        cost = conn.execute(
            "SELECT coalesce(sum(cost_usd), 0) AS usd FROM usage_ledger "
            "WHERE recorded_at >= %s AND kind IN ('page_reading', 'transcription')",
            (since,),
        ).fetchone()["usd"]
    reread = sum(result.status == "ingested" for result in results)
    return Ingested(len(paths), reread, chunks, seconds, float(cost))


def run_perry(questions: list[Question]) -> None:
    ingest_perry()
    from perry.agents import knowledge

    answer_all(questions, knowledge.ask, answers_path("perry"))


@dataclass(frozen=True)
class RetrievalRun:
    """Page recall of hybrid search over the benchmark Documents as one Reader and
    Chunking Strategy stored them: the Citations each question's top hits point at."""

    reader: str
    chunking: str
    embedding_model: str
    ingested: Ingested
    plain: dict[str, set[str]]
    reranked: dict[str, set[str]] | None

    @property
    def setup(self) -> str:
        return f"{self.reader} reader, {self.chunking} chunking, {self.embedding_model}"


def run_perry_retrieval(
    questions: list[Question], chunking: str, reader: str, limit: int = 6
) -> RetrievalRun:
    """Page recall of hybrid search alone: which required pages are in the top hits, as
    hybrid search ranks them and, if there is a reranker, as it reranks a wider pool.
    The Documents are read with `reader` and chunked with `chunking` first."""
    ingested = ingest_perry(chunking, reader)
    from perry import models, techniques
    from perry.search import hybrid_search

    with models.without_reranker():
        plain = {
            q.id: {hit.citation for hit in hybrid_search(q.question, limit=limit)}
            for q in questions
        }
    reranked = None
    if models.reranking_supported():
        hybrid = techniques.get("hybrid_search")
        reranked = {
            q.id: {hit.citation for hit in techniques.search(hybrid, q.question).hits}
            for q in questions
        }
    return RetrievalRun(reader, chunking, models.embedding_model(), ingested, plain, reranked)


def _recall(found: set[str], q: Question) -> float:
    return len(found & set(q.pages)) / len(q.pages)


def comparison_report(questions: list[Question], runs: list[RetrievalRun], limit: int = 6) -> str:
    """One row per Reader and Chunking Strategy: page recall over every question, for
    hybrid search's own top hits and reranked, with what reading the Documents took."""
    lines = [
        "# Perry retrieval: Readers and Chunking Strategies compared (no chat model)",
        "",
        (
            f"Share of required source pages among the top {limit} hits, over all "
            f"{len(questions)} benchmark questions. Ingest time counts reading, chunking and "
            "embedding the Documents (- if they were already stored that way). Reading cost "
            "is what the vision model charged to read pages."
        ),
        "",
        (
            "| Reader | Chunking | Chunks | Ingest time | Reading cost "
            "| Hybrid search: page recall | Hybrid search: all pages found "
            "| Reranked: page recall | Reranked: all pages found |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        cells = []
        for found in (run.plain, run.reranked):
            if found is None:
                cells += ["-", "-"]
                continue
            recalls = [_recall(found[q.id], q) for q in questions]
            cells += [_pct(_mean(recalls)), f"{sum(r == 1.0 for r in recalls)}/{len(questions)}"]
        ingested = run.ingested
        took = f"{ingested.seconds:.0f} s" if ingested.reread == ingested.documents else "-"
        lines.append(
            f"| {run.reader} | {run.chunking} | {ingested.chunks} | {took} "
            f"| ${ingested.reading_cost_usd:.4f} | {' | '.join(cells)} |"
        )
    return "\n".join(lines) + "\n"


def retrieval_report(
    questions: list[Question],
    limit: int,
    plain: dict[str, set[str]],
    reranked: dict[str, set[str]] | None,
    setup: str = "",
) -> str:
    """The retrieval-only report: page recall per question type and per question, for
    hybrid search's own top hits and, if given, the reranked ones. `setup` names what was
    measured, e.g. the Chunking Strategy and embedding model."""
    runs = [("Hybrid search", plain)] + ([("Reranked", reranked)] if reranked else [])
    recall = _recall

    lines = [
        "# Perry retrieval only (no chat model)" + (f": {setup}" if setup else ""),
        "",
        (
            f"Share of required source pages among the top {limit} hits for the question as "
            "asked: hybrid search's own top hits"
            + (
                f", and the top {limit} the reranker picks from a wider pool of hybrid "
                "search's hits"
                if reranked
                else ""
            )
            + ". The Knowledge Agent may rephrase and search up to 3 times, so this is a "
            "lower bound on what it sees."
        ),
        "",
        "| Type | Questions | "
        + " | ".join(f"{name}: page recall | {name}: all pages found" for name, _ in runs)
        + " |",
        "|---|---|" + "---|---|" * len(runs),
    ]
    for qtype in (*QUESTION_TYPES, "all"):
        group = [q for q in questions if qtype in ("all", q.type)]
        if not group:
            continue
        cells = []
        for _, found in runs:
            recalls = [recall(found[q.id], q) for q in group]
            cells += [
                _pct(_mean(recalls)),
                f"{sum(r == 1.0 for r in recalls)}/{len(group)}",
            ]
        lines.append(f"| {qtype} | {len(group)} | {' | '.join(cells)} |")
    lines += [
        "",
        "| Question | "
        + " | ".join(f"{name}: page recall | {name}: missing pages" for name, _ in runs)
        + " |",
        "|---|" + "---|---|" * len(runs),
    ]
    for q in questions:
        cells = []
        for _, found in runs:
            missing = ", ".join(p for p in q.pages if p not in found[q.id]) or "-"
            cells += [_pct(recall(found[q.id], q)), missing]
        lines.append(f"| {q.id} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def _qabot(*args: str, input: str | None = None) -> str:
    python = QABOT_DIR / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = QABOT_DIR / ".venv" / "bin" / "python"
    env = {
        **os.environ,
        "PYTHONPATH": str(QABOT_DIR),
        "RAG_STORAGE_DIR": str(STORAGE_DIR / "qabot"),
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [str(python), str(HERE / "qabot_adapter.py"), *args],
        input=input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=QABOT_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"QA-bot failed:\n{result.stderr[-3000:]}")
    return result.stdout


def ask_qabot(question: str) -> str:
    # The JSON answer is the last stdout line; anything before it is library chatter.
    output = _qabot("ask", QABOT_MODE, input=question)
    return json.loads(output.strip().splitlines()[-1])["answer"]


def run_qabot(questions: list[Question]) -> None:
    print(_qabot("ingest", str(DOCUMENTS_DIR)), flush=True)
    answer_all(questions, ask_qabot, answers_path("qabot"))


def compare_retrieval(
    questions: list[Question],
    readers: list[str] | None,
    chunkings: list[str] | None,
    *,
    yes: bool = False,
    ask=input,
) -> list[RetrievalRun]:
    """`perry-retrieval` for every Reader and Chunking Strategy named (PERRY_DOCUMENT_READER
    and PERRY_CHUNKING if None), each in turn, writing each one's report and, for more than
    one, results/perry-retrieval-comparison.md. The vision Reader calls a paid model, so
    its estimate is shown and an OK awaited (unless `yes`) first."""
    _use_benchmark_database()
    from perry import db, documents, models, usage

    readers = readers or [documents.READER]
    chunkings = chunkings or [documents.CHUNKING]
    for kind, named, allowed in (
        ("reader", readers, documents.READERS),
        ("chunking", chunkings, documents.CHUNKINGS),
    ):
        for name in named:
            if name not in allowed:
                sys.exit(f"Unknown {kind} {name!r}: use {', '.join(allowed)}.")
    db.init_schema()
    if "vision" in readers:
        if not models.page_reader_supported():
            sys.exit(
                f"Not started: the vision reader runs on {models.PAGE_READER_MODEL} and "
                "OPENAI_API_KEY isn't set."
            )
        pages = sum(
            len(documents.parse(path, chunkings[0], "vision").pages_to_read)
            for path in sorted(DOCUMENTS_DIR.glob("*.pdf"))
        )
        reads = pages * len(chunkings)  # every Chunking Strategy reads the pages again
        cost = reads * usage.cost(
            models.page_reader_model(),
            models.PAGE_TRANSCRIPTION_INPUT_TOKENS,
            models.PAGE_TRANSCRIPTION_OUTPUT_TOKENS,
        )
        print(
            f"The vision reader reads {reads} pages on {models.page_reader_model()}: "
            f"about ${cost:.2f}.",
            flush=True,
        )
        if not yes and ask("Go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("Not started.")
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = []
    for reader in readers:
        for chunking in chunkings:
            run = run_perry_retrieval(questions, chunking, reader)
            report = retrieval_report(questions, 6, run.plain, run.reranked, run.setup)
            path = RESULTS_DIR / f"perry-retrieval-{reader}-{chunking}.md"
            path.write_text(report, encoding="utf-8")
            runs.append(run)
    if len(runs) > 1:
        report = comparison_report(questions, runs)
        (RESULTS_DIR / "perry-retrieval-comparison.md").write_text(report, encoding="utf-8")
    print(report)
    return runs


def run_grid(
    questions: list[Question],
    only: list[str] | None,
    rerun: bool,
    *,
    yes: bool = False,
    ask=input,
) -> None:
    """The technique grid (see benchmark/grid.py) in the benchmark's own database, every
    Turn on the one paid model (PERRY_BENCHMARK_LLM_MODEL) so rows never mix two models.
    Shows an estimate and waits for an OK (unless `yes`) before any paid call. The table
    is redrawn even when the run stops early, so what was answered is shown."""
    _use_benchmark_database()
    from benchmark import grid
    from perry import models

    if not models.paid_chat_available():
        sys.exit("Not started: the grid runs on the paid model and OPENAI_API_KEY isn't set.")
    setups = grid.select(only)
    path = RESULTS_DIR / "grid.jsonl"
    try:
        grid.prepare(setups, sorted(DOCUMENTS_DIR.glob("*.pdf")))
    except grid.NotReady as not_ready:
        sys.exit(f"Not started: {not_ready}")
    turns = grid.turns_to_ask(questions, setups, path, rerun)
    model = models.BENCHMARK_LLM_MODEL
    print(
        f"{turns} Turns to ask on {model}: about ${grid.estimate_usd(turns, model):.2f} "
        f"(assuming {grid.TURN_INPUT_TOKENS:,} input and {grid.TURN_OUTPUT_TOKENS:,} output "
        "tokens a Turn).",
        flush=True,
    )
    if turns and not yes and ask("Go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
        sys.exit("Not started.")
    try:
        with models.paid_chat_only(model):
            grid.run_grid(questions, setups, path, grid.ask_perry, rerun=rerun)
    finally:
        print(write_grid_report(questions).split("## Correct per")[0])


def write_grid_report(questions: list[Question]) -> str:
    from benchmark import grid

    report = grid.render_grid(questions, grid.load_grid(RESULTS_DIR / "grid.jsonl"))
    (RESULTS_DIR / "GRID.md").write_text(report, encoding="utf-8")
    return report


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="benchmark.run", usage=__doc__)
    parser.add_argument(
        "command", choices=("perry", "qabot", "perry-retrieval", "report", "grid", "grid-report")
    )
    parser.add_argument(
        "--only",
        help="grid: comma-separated setups (technique names, semantic, logical, cascade, "
        "always-hybrid), each optionally with +rerank (e.g. cascade+rerank); the rest of "
        "the grid's rows are kept",
    )
    parser.add_argument("--rerun", action="store_true", help="grid: ask the selected setups afresh")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="grid, perry-retrieval: start without asking about the estimate",
    )
    parser.add_argument(
        "--reader",
        help="perry-retrieval: comma-separated Readers to compare (docling, pymupdf, vision; "
        "default PERRY_DOCUMENT_READER)",
    )
    parser.add_argument(
        "--chunking",
        help="perry-retrieval: comma-separated Chunking Strategies to compare (heading, "
        "fixed; default PERRY_CHUNKING). Each combination's report goes to "
        "results/perry-retrieval-<reader>-<chunking>.md, and with more than one, a summary "
        "to results/perry-retrieval-comparison.md",
    )
    args = parser.parse_args(argv)
    command = args.command
    questions = load_questions(QUESTIONS_PATH)
    if command == "grid":
        only = [name.strip() for name in args.only.split(",")] if args.only else None
        run_grid(questions, only, args.rerun, yes=args.yes)
    elif command == "grid-report":
        print(write_grid_report(questions))
    elif command == "perry":
        run_perry(questions)
    elif command == "qabot":
        run_qabot(questions)
    elif command == "perry-retrieval":

        def names(value: str | None) -> list[str] | None:
            return [name.strip() for name in value.split(",")] if value else None

        compare_retrieval(questions, names(args.reader), names(args.chunking), yes=args.yes)
    elif command == "report":
        answers = {system: load_answers(answers_path(system)) for system in SYSTEMS}
        report = render_report(questions, answers)
        (RESULTS_DIR / "RESULTS.md").write_text(report, encoding="utf-8")
        print(report.split("## Per question\n")[0])


if __name__ == "__main__":
    main(sys.argv[1:])
