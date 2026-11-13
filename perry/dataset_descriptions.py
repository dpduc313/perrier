"""Descriptions of Datasets and columns the Data Dictionary leaves out, written in the
background (ADR-0008), as Summaries are for Documents (ADR-0003), by the model
PERRY_DATASET_DESCRIPTION_MODEL names: the company's model or the local summary model
(`models.dataset_description_model`).

An upload returns at once, with the Dataset's description status `pending` while it or
any column has no description. Then, one Dataset at a time, the job asks that model for
one sentence per undescribed column and, if it has none, one for the Dataset, from
names, types and Value Profiles only. Its descriptions are marked as the
model's (`DescriptionSource.MODEL`) and fill only what is still undescribed when they
arrive, so they never overwrite the Owner's (`datasets.resolve_description`). A Dataset
whose describing fails, or whose reply leaves out something asked for, stays usable, is
marked `failed`, and is retried on the next start (`resume()`).
"""

import json
import logging
import re
import threading
from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from perry import db, models
from perry.datasets import (
    ITSELF,
    Column,
    DatasetItself,
    Description,
    DescriptionSource,
    DescriptionStatus,
    resolve_description,
    save_descriptions,
)

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = """You describe the columns of a table so an analyst knows what they hold.
You see only the table's name and, for each column, its name, type and what its values \
look like. Reply with only a JSON object with the keys you are asked for:
- "columns": an object mapping each column you are asked about to one sentence on what \
it holds, with its unit when the values show it.
- "dataset", only when you are asked what one row is: one sentence saying what one row \
of the table is.
Write in the language of the names. Say only what the names and values support."""

# What the model is asked to describe: column names, and ITSELF for the Dataset.
Asked = str | DatasetItself


def describe_pending() -> int:
    """Describe every pending Dataset, one at a time, and return how many were tried.
    Each ends `ready` or, if the model fails, `failed`. Without the model that describes
    Datasets (`models.dataset_description_model`), they stay pending."""
    if not models.dataset_descriptions_supported():
        return 0
    tried = 0
    while (name := _next_pending()) is not None:
        tried += 1
        try:
            _describe(name)
        except Exception:
            log.warning("Couldn't describe %s; will retry on the next start", name, exc_info=True)
            _set_status(name, DescriptionStatus.FAILED)
    return tried


def _next_pending() -> str | None:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT name FROM datasets WHERE description_status = %s ORDER BY name LIMIT 1",
            (str(DescriptionStatus.PENDING),),
        ).fetchone()
    return row["name"] if row else None


def _set_status(name: str, status: DescriptionStatus) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE datasets SET description_status = %s WHERE name = %s", (str(status), name)
        )


def _describe(name: str) -> None:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT columns, description FROM datasets WHERE name = %s", (name,)
        ).fetchone()
    columns = list(map(Column.from_json, row["columns"]))
    wanted = [c for c in columns if c.description is None]
    itself = row["description"] is None
    asked: set[Asked] = {c.name for c in wanted} | ({ITSELF} if itself else set())
    if asked:
        # The slow model call happens outside any transaction.
        reply = models.dataset_description_model().invoke(
            [SystemMessage(DESCRIBE_PROMPT), HumanMessage(_request(name, columns, itself))]
        )
        guesses = parse_reply(reply.text, asked)
    else:
        guesses = {}
    _save(name, guesses, asked)


def _request(name: str, columns: list[Column], itself: bool) -> str:
    """What the model reads: the Dataset's name and the undescribed columns' names, types
    and Value Profiles (and, when the Dataset itself is to be described, the others'
    names and types). A Dataset already described isn't mentioned at all: a model told to
    leave a key empty may echo that line instead of the JSON object."""
    lines = [f"Table: {name}", "Describe these columns:"]
    for column in columns:
        if column.description is None:
            profile = column.value_profile
            values = profile.describe() if profile else "values unknown"
            lines.append(f"- {column.name} ({column.type}): {values}")
    others = [f"{c.name} ({c.type})" for c in columns if c.description is not None]
    if itself:
        if others:
            lines.append(f"Its other columns: {', '.join(others)}")
        lines.append("Also say what one row of the table is.")
        lines.append('Reply with "columns" and "dataset".')
    else:
        lines.append('Reply with "columns".')
    return "\n".join(lines)


def parse_reply(text: str, asked: set[Asked]) -> dict[Asked, str]:
    """The model's sentences for what it was asked about, keyed by column name (ITSELF for
    the Dataset). A reply that is just the columns' object counts too. Raises ValueError
    when the reply isn't a JSON object."""
    data = _json_object(text)
    if data is None:
        raise ValueError(f"The model's reply isn't a JSON object: {text[:200]!r}")
    columns = data.get("columns")
    found: dict[Asked, object] = dict(columns if isinstance(columns, dict) else data)
    found[ITSELF] = data.get("dataset")
    return {
        key: str(value).strip()
        for key, value in found.items()
        if key in asked and value and str(value).strip()
    }


def _json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)  # tolerates ```json fences and chatter
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _save(name: str, guesses: dict[Asked, str], asked: set[Asked]) -> None:
    """Fill what is still undescribed with the model's guesses. What the model was asked
    about but left out of its reply makes the run failed, to be retried on the next start;
    what it wasn't asked about (a column a re-upload added meanwhile) stays pending."""
    with db.pool().connection() as conn, conn.transaction():
        row = conn.execute(
            "SELECT columns, description, description_source FROM datasets"
            " WHERE name = %s FOR UPDATE",
            (name,),
        ).fetchone()
        if row is None:  # removed meanwhile
            return
        columns = [
            replace(c, description=_with_guess(c.description, guesses.get(c.name)))
            for c in map(Column.from_json, row["columns"])
        ]
        description = _with_guess(Description.from_row(row), guesses.get(ITSELF))
        left: set[Asked] = {c.name for c in columns if c.description is None}
        if description is None:
            left.add(ITSELF)
        if left & asked:  # the reply left something out: retried on the next start
            status = DescriptionStatus.FAILED
        elif left:  # a column a re-upload added meanwhile
            status = DescriptionStatus.PENDING
        else:
            status = DescriptionStatus.READY
        save_descriptions(conn, name, columns, description, status)


def _with_guess(current: Description | None, guess: str | None) -> Description | None:
    """The description once the model's guess, if any, is offered: the Owner's, or an
    earlier guess, stays."""
    if guess is None:
        return current
    return resolve_description(current, Description(guess, DescriptionSource.MODEL))


# --- Background job -------------------------------------------------------------

_lock = threading.Lock()
_wake = threading.Event()
_thread: threading.Thread | None = None


def start_background() -> None:
    """Describe pending Datasets on a background thread, without waiting. Called after
    an upload; if the job is already running, it makes another pass when it's done.
    Without the model that describes Datasets there is nothing to run."""
    global _thread
    if not models.dataset_descriptions_supported():
        return
    with _lock:
        _wake.set()
        if _thread is None:
            _thread = threading.Thread(target=_run, name="perry-dataset-descriptions", daemon=True)
            _thread.start()


def _run() -> None:
    global _thread
    while True:
        with _lock:
            if not _wake.is_set():
                _thread = None
                return
            _wake.clear()
        try:
            describe_pending()
        except Exception:
            log.exception("Background describing of Datasets stopped")


def resume() -> None:
    """On start: retry the Datasets whose describing failed, and finish pending ones, in
    the background."""
    retry_failed()
    start_background()


def retry_failed() -> None:
    """Make the Datasets whose describing failed pending again."""
    with db.pool().connection() as conn:
        conn.execute(
            "UPDATE datasets SET description_status = %s WHERE description_status = %s",
            (str(DescriptionStatus.PENDING), str(DescriptionStatus.FAILED)),
        )
