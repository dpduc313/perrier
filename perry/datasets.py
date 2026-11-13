"""Datasets: CSV files and spreadsheet sheets stored as tables in the analytics schema.

Uploading writes through the main connection. Querying (`run_query`) connects as the
read-only analyst role, so whatever SQL the Analysis Agent writes, Postgres only lets
it run one statement that reads Datasets, within a statement timeout, and Perry
returns at most a row limit.

Each Dataset's registry entry also carries descriptions, Value Profiles and Join Hints
(ADR-0008). `upload()` is the one way in: it hands a Data Dictionary to
perry.data_dictionary, and checks Join Hints through perry.join_hints. Those two build on
this module's types, so this module imports them where it calls them.
"""

import os
import re
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from perry import db

SUPPORTED_SUFFIXES = {".csv", ".xlsx"}
SCHEMA = "analytics"

STATEMENT_TIMEOUT_MS = int(os.getenv("PERRY_ANALYSIS_TIMEOUT_MS", "10000"))
ROW_LIMIT = int(os.getenv("PERRY_ANALYSIS_ROW_LIMIT", "200"))

# Postgres identifiers are at most 63 bytes.
_MAX_IDENTIFIER = 63


# A text column with at most this many distinct values is profiled with all of them;
# one with more, with a few examples.
MAX_LISTED_VALUES = 20
EXAMPLES = 3
# Examples are cut to this many characters, so a column of long notes stays short. A short
# list's values are shown in full, so the Agent can filter on them exactly.
_MAX_EXAMPLE_CHARS = 60
# Text that may be an ISO date or timestamp: 2026-03, 2026-03-31, 2026-03-31T08:00, ...
_ISO_DATE = re.compile(r"\d{4}-\d{2}(-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?)?")


@dataclass(frozen=True)
class ValueProfile:
    """What a column's values look like, computed from the rows at upload."""

    empty_share: float
    min: int | float | str | None = None  # numbers, timestamps and ISO-date text
    max: int | float | str | None = None
    values: list | None = None  # every distinct value, when there are few
    examples: list | None = None  # a few values, when there are many
    distinct: int | None = None  # how many distinct values, when only examples are kept

    @classmethod
    def of(cls, series: pd.Series) -> "ValueProfile":
        present = series.dropna()
        empty = 1 - len(present) / len(series) if len(series) else 0.0
        if present.empty:
            return cls(empty)
        if pd.api.types.is_bool_dtype(series):
            return cls(empty, values=sorted({bool(v) for v in present}))
        if pd.api.types.is_numeric_dtype(series):
            return cls(empty, min=_plain(present.min()), max=_plain(present.max()))
        if pd.api.types.is_datetime64_any_dtype(series):
            return cls(empty, min=present.min().isoformat(), max=present.max().isoformat())
        text = present.astype(str)
        low = high = None
        if text.map(_ISO_DATE.fullmatch).all():
            dates = pd.to_datetime(text, format="ISO8601", errors="coerce")
            if dates.notna().all():
                low, high = text[dates.idxmin()], text[dates.idxmax()]
        distinct = list(dict.fromkeys(text))
        if len(distinct) <= MAX_LISTED_VALUES:
            return cls(empty, low, high, values=sorted(distinct))
        return cls(empty, low, high, examples=distinct[:EXAMPLES], distinct=len(distinct))

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_json(cls, data: dict) -> "ValueProfile":
        return cls(**data)

    def describe(self) -> str:
        if self.empty_share >= 1:
            return "all empty"
        parts = [_share_empty(self.empty_share)]
        if self.min is not None:
            parts.append(f"min {self.min}, max {self.max}")
        if self.values is not None:
            parts.append(f"values {_listed(self.values)}")
        if self.examples is not None:
            examples = _listed(self.examples, cut=True)
            parts.append(f"{self.distinct} distinct values, e.g. {examples}")
        return "; ".join(parts)


def _plain(number) -> int | float:
    """A number as JSON stores it, and whole floats as the integers they read as."""
    number = number.item() if hasattr(number, "item") else number
    return int(number) if isinstance(number, float) and number.is_integer() else number


def _share_empty(share: float) -> str:
    if share == 0:
        return "no empty values"
    return "<1% empty" if share < 0.01 else f"{share:.0%} empty"


def _listed(values: list, cut: bool = False) -> str:
    """Values as SQL literals, so the Agent can filter on them exactly as written. `cut`:
    shorten long ones (examples, which the Agent doesn't filter on)."""
    shown = []
    for value in values:
        if isinstance(value, bool):
            shown.append(str(value).lower())
            continue
        text = value
        if cut and len(value) > _MAX_EXAMPLE_CHARS:
            text = value[:_MAX_EXAMPLE_CHARS] + "…"
        shown.append("'" + text.replace("'", "''") + "'")
    return ", ".join(shown)


class DescriptionSource(StrEnum):
    """Who wrote a description of a Dataset or column."""

    USER = "user"  # the Owner, in a Data Dictionary
    MODEL = "model"  # the model that describes Datasets, as a guess (perry.dataset_descriptions)


class DescriptionStatus(StrEnum):
    """Whether the model has described what has no description yet."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"  # retried on the next start


class JoinHintSource(StrEnum):
    INFERRED = "inferred"  # from the column's name and values
    DECLARED = "declared"  # by a Data Dictionary's `references`


class DatasetItself(Enum):
    """Stands for the Dataset itself where its columns are named: a Data Dictionary row
    with an empty `column`, or what the model is asked to describe."""

    ITSELF = "the Dataset itself"


ITSELF = DatasetItself.ITSELF


@dataclass(frozen=True)
class Description:
    """What a Dataset or column holds, and who said so."""

    text: str
    source: DescriptionSource

    @classmethod
    def from_row(cls, row) -> "Description | None":
        """The description in a registry row or column entry, if it has one."""
        if row.get("description") is None:
            return None
        return cls(row["description"], DescriptionSource(row["description_source"]))


def resolve_description(
    current: Description | None, given: Description | None
) -> Description | None:
    """The one rule for who describes a Dataset or column: the Owner beats the model.

    - The Owner's description replaces any other.
    - A model guess fills only what has no description, so it never overwrites the
      Owner's (or an earlier guess).
    - `given` None: a Data Dictionary that names the Dataset leaves this out, which takes
      the Owner's earlier description away and leaves a guess.
    """
    if given is None:
        return None if current and current.source == DescriptionSource.USER else current
    if given.source == DescriptionSource.USER or current is None:
        return given
    return current


@dataclass(frozen=True)
class JoinHint:
    """That the column holds values of another Dataset's column."""

    dataset: str
    column: str
    source: JoinHintSource = JoinHintSource.INFERRED
    orphans: int = 0  # values not found in the target (only a declared hint keeps any)

    def to_json(self) -> dict:
        data = {"dataset": self.dataset, "column": self.column, "source": str(self.source)}
        if self.orphans:
            data["orphans"] = self.orphans
        return data

    @classmethod
    def from_json(cls, data: dict) -> "JoinHint":
        return cls(
            data["dataset"], data["column"], JoinHintSource(data["source"]), data.get("orphans", 0)
        )

    def target(self) -> str:
        return f"{self.dataset}.{self.column}"

    def describe(self) -> str:
        n = self.orphans
        orphans = f", {n} value{'s' if n != 1 else ''} not found there" if n else ""
        return f"references {self.target()} ({self.source}{orphans})"


@dataclass(frozen=True)
class Column:
    """One column of a Dataset as the Analysis Agent learns it.

    In the registry's JSON the Value Profile is `profile` and the Join Hint `references`,
    the keys ADR-0008's spec gave them."""

    name: str
    type: str
    value_profile: ValueProfile | None  # None for a Dataset uploaded before Value Profiles
    description: Description | None = None
    join_hint: JoinHint | None = None

    def to_json(self) -> dict:
        data = {"name": self.name, "type": self.type}
        if self.value_profile is not None:
            data["profile"] = self.value_profile.to_json()
        if self.description is not None:
            data["description"] = self.description.text
            data["description_source"] = str(self.description.source)
        if self.join_hint is not None:
            data["references"] = self.join_hint.to_json()
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Column":
        profile, hint = data.get("profile"), data.get("references")
        return cls(
            data["name"],
            data["type"],
            profile and ValueProfile.from_json(profile),
            Description.from_row(data),
            hint and JoinHint.from_json(hint),
        )

    def describe(self, pending: bool = False) -> str:
        """The column's line in describe_datasets. `pending`: the model hasn't
        described the Dataset's undescribed columns yet."""
        parts = [
            self.value_profile.describe()
            if self.value_profile
            else "no value profile yet (upload it again)"
        ]
        if self.join_hint:
            parts.append(self.join_hint.describe())
        facts = "; ".join(parts)
        described = marked(self.description, pending)
        if described:
            return f"- {self.name} ({self.type}): {described} [{facts}]"
        return f"- {self.name} ({self.type}): {facts}"


def sentence(text: str) -> str:
    """Text ending like a sentence, so whatever follows it reads apart."""
    return text if text[-1] in ".!?" else f"{text}."


NOT_YET = "No description yet."
GUESS_MARK = "(the model's guess)"


def marked(description: Description | None, pending: bool = False) -> str:
    """A description as the Agent and the Owner read it: the model's marked as a guess,
    and a missing one as not written yet while the model may still write it."""
    if not description:
        return NOT_YET if pending else ""
    if description.source == DescriptionSource.MODEL:
        return f"{sentence(description.text)} {GUESS_MARK}"
    return sentence(description.text)


@dataclass(frozen=True)
class Dataset:
    name: str  # also the table name the SQL uses
    origin: str  # the uploaded file it came from
    column_details: list[Column]
    row_count: int
    description: Description | None = None  # what one row is
    # PENDING until the model has described what the Data Dictionary left out
    description_status: DescriptionStatus = DescriptionStatus.READY

    @property
    def columns(self) -> list[str]:
        return [c.name for c in self.column_details]

    @property
    def types(self) -> list[str]:
        return [c.type for c in self.column_details]

    @property
    def pending(self) -> bool:
        """Whether the model may still describe what has no description."""
        return self.description_status != DescriptionStatus.READY

    def describe(self) -> str:
        """One compact line: what list_datasets shows."""
        cols = ", ".join(f"{c} ({t})" for c, t in zip(self.columns, self.types, strict=True))
        described = marked(self.description, self.pending)
        if described:
            cols = f"{described} Columns: {cols}"
        hints = ", ".join(
            f"{c.name} → {c.join_hint.target()}" for c in self.column_details if c.join_hint
        )
        joins = f"; join hints: {hints}" if hints else ""
        return f"{self.name} ({self.row_count} rows, from {self.origin}): {cols}{joins}"

    def describe_columns(self) -> str:
        """The Dataset's description, then every column with its type, description and
        Value Profile: what describe_datasets shows."""
        lines = [f"{self.name} ({self.row_count} rows, from {self.origin})"]
        described = marked(self.description, self.pending)
        if described:
            lines.append(described)
        return "\n".join([*lines, *(c.describe(self.pending) for c in self.column_details)])

    def describe_for_panel(self) -> str:
        """What the Datasets panel shows after the name: the description, the column
        names, and which column descriptions are the model's guesses."""
        line = f"Columns: {', '.join(self.columns)}"
        described = marked(self.description, self.pending)
        if described:
            line = f"{described} {line}"
        guessed = [
            c.name
            for c in self.column_details
            if c.description and c.description.source == DescriptionSource.MODEL
        ]
        if guessed:
            line += f"; the model guessed the descriptions of {', '.join(guessed)}"
        return line


@dataclass(frozen=True)
class UploadResult:
    dataset: Dataset
    replaced: bool
    # Data Dictionary rows that waited for this Dataset but name a column it lacks
    not_attached: list[str] = field(default_factory=list)
    # declared Join Hints, any Dataset's, the data no longer supports
    join_hints_dropped: list[str] = field(default_factory=list)
    # declared Join Hints, any Dataset's, kept with values not found in their target
    join_hints_orphaned: list[str] = field(default_factory=list)
    # declared Join Hints of this Dataset whose target Dataset isn't uploaded yet
    join_hints_waiting: list[str] = field(default_factory=list)

    def describe(self) -> str:
        d = self.dataset
        verb = "replaced" if self.replaced else "created"
        line = f"{d.name}: {verb}, {d.row_count} rows, columns {', '.join(d.columns)}"
        itself = _by_owner(d.description)
        columns = sum(_by_owner(c.description) for c in d.column_details)
        if itself or columns:
            line += f"; the Owner's descriptions on {count_described(itself, columns)}"
        itself = d.description is None
        columns = sum(c.description is None for c in d.column_details)
        if itself or columns:
            line += f"; left for the model to describe: {count_described(itself, columns)}"
        if self.not_attached:
            line += f"; not attached: {', '.join(self.not_attached)}"
        # "references", as the Owner wrote them in a Data Dictionary's `references` column
        for label, hints in (
            ("declared references dropped", self.join_hints_dropped),
            ("values not found", self.join_hints_orphaned),
            ("declared references waiting", self.join_hints_waiting),
        ):
            if hints:
                line += f"; {label}: {', '.join(hints)}"
        return line


def _by_owner(description: Description | None) -> bool:
    return description is not None and description.source == DescriptionSource.USER


def count_described(itself: bool, columns: int) -> str:
    """'the Dataset and 2 columns': what is described in one Dataset."""
    what = [f"{columns} column{'s' if columns != 1 else ''}"] if columns else []
    if itself:
        what.insert(0, "the Dataset")
    return " and ".join(what)


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool  # more rows existed than the row limit


class QueryError(Exception):
    """A query that failed or timed out, with a message fit to show the user."""


def identifier(text: str, fallback: str) -> str:
    """A plain lowercase SQL identifier, so the model can write it without quoting."""
    name = re.sub(r"\W+", "_", text.lower()).strip("_") or fallback
    if name[0].isdigit():
        name = f"{fallback}_{name}"
    return name[:_MAX_IDENTIFIER]


def _column_names(labels) -> list[str]:
    names: list[str] = []
    for label in labels:
        base = identifier(str(label), "column")
        name, n = base, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.append(name)
    return names


def _sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    return "text"


def _python(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def upload(path: Path) -> list:
    """Store each sheet of the file as a Dataset, replacing any Dataset of the same name,
    and return one result per Dataset or Data Dictionary, each with `describe()`.

    A Data Dictionary (a `.dictionary.csv` file, or a workbook's `dictionary` sheet)
    describes Datasets instead of being one. A refused `.dictionary.csv` raises
    data_dictionary.DictionaryRefused; a workbook's refused `dictionary` sheet is reported
    in its result, and the workbook's other sheets are stored all the same.
    """
    from perry import data_dictionary

    suffix = path.suffix.lower()
    if suffix == ".csv" and path.stem.lower().endswith(data_dictionary.SUFFIX):
        default = identifier(path.stem[: -len(data_dictionary.SUFFIX)], "dataset")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        return [data_dictionary.apply(path.name, data_dictionary.parse(frame, default=default))]
    if suffix == ".csv":
        return [_store(identifier(path.stem, "dataset"), path.name, pd.read_csv(path))]
    if suffix != ".xlsx":
        raise ValueError(f"Unsupported file type: {path.name} (use .csv or .xlsx)")
    sheets = pd.read_excel(path, sheet_name=None)
    dictionaries = [s for s in sheets if str(s).strip().lower() == data_dictionary.SHEET]
    names = {
        sheet: identifier(f"{path.stem} {sheet}", "dataset")
        for sheet in sheets
        if sheet not in dictionaries
    }
    by_sheet = {identifier(str(sheet), "dataset"): name for sheet, name in names.items()}
    parsed, refused = [], []
    for sheet in dictionaries:
        try:
            parsed.append(data_dictionary.parse(sheets[sheet], sheets=by_sheet))
        except data_dictionary.DictionaryRefused as error:  # the data sheets still upload
            refused.append(data_dictionary.RefusedDictionary(path.name, str(error)))
    stored = [_store(name, path.name, sheets[sheet]) for sheet, name in names.items()]
    applied = [data_dictionary.apply(path.name, rows) for rows in parsed]
    if applied:  # each sheet's reply says what the workbook's dictionary described
        now = {d.name: d for d in list_datasets()}
        stored = [replace(r, dataset=now[r.dataset.name]) for r in stored]
    return [*stored, *applied, *refused]


def _store(name: str, origin: str, frame: pd.DataFrame) -> UploadResult:
    from perry import data_dictionary, join_hints

    columns = _column_names(frame.columns)
    types = [_sql_type(frame[c]) for c in frame.columns]
    details = [
        Column(name, type_, ValueProfile.of(frame.iloc[:, i]))
        for i, (name, type_) in enumerate(zip(columns, types, strict=True))
    ]
    table = sql.Identifier(SCHEMA, name)
    with db.pool().connection() as conn, conn.transaction():
        old = conn.execute(
            "SELECT columns, description, description_source FROM datasets"
            " WHERE name = %s FOR UPDATE",
            (name,),
        ).fetchone()
        description = None
        if old:  # columns that survive by name keep their descriptions and declared hints
            if _by_owner(kept := Description.from_row(old)):  # a guess goes back to the model
                description = kept
            namesakes = {c["name"]: Column.from_json(c) for c in old["columns"]}
            details = [_carried_over(c, namesakes.get(c.name)) for c in details]
        not_attached = []
        attached = data_dictionary.attach_waiting_rows(conn, name, details, description)
        if attached:
            details, description = attached.columns, attached.description
            not_attached = attached.not_attached
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
        conn.execute(
            sql.SQL("CREATE TABLE {} ({})").format(
                table,
                sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(t))
                    for c, t in zip(columns, types, strict=True)
                ),
            )
        )
        copy = sql.SQL("COPY {} ({}) FROM STDIN").format(
            table, sql.SQL(", ").join(map(sql.Identifier, columns))
        )
        with conn.cursor().copy(copy) as writer:
            for row in frame.itertuples(index=False, name=None):
                writer.write_row([_python(v) for v in row])
        conn.execute(
            """
            INSERT INTO datasets (name, origin, columns, row_count) VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET origin = excluded.origin,
                columns = excluded.columns, row_count = excluded.row_count, uploaded_at = now()
            """,
            (name, origin, Jsonb([c.to_json() for c in details]), len(frame)),
        )
        status = description_status(description, details)
        save_descriptions(conn, name, details, description, status)
        # Declared hints waiting for their target: rows that waited for this Dataset
        # replace them, and otherwise those of columns that survive wait on.
        join_hints.forget_waiting_except(conn, name, [] if attached else columns)
        join_hints.attach_waiting(conn, name)
        # checks the declared hints carried over, claimed or attached above against the
        # new rows
        checked = join_hints.recheck(conn, name)
    dataset = Dataset(name, origin, checked.columns, len(frame), description, status)
    return UploadResult(
        dataset,
        old is not None,
        not_attached,
        checked.dropped,
        checked.orphaned,
        checked.waiting,
    )


def description_status(description: Description | None, columns: list[Column]) -> DescriptionStatus:
    """PENDING while the Dataset or a column has no description for the model to fill."""
    if description is None or any(c.description is None for c in columns):
        return DescriptionStatus.PENDING
    return DescriptionStatus.READY


def save_descriptions(
    conn: psycopg.Connection,
    name: str,
    columns: list[Column],
    description: Description | None,
    status: DescriptionStatus,
) -> None:
    """Store a Dataset's columns, its own description and its description status."""
    conn.execute(
        "UPDATE datasets SET columns = %s, description = %s, description_source = %s,"
        " description_status = %s WHERE name = %s",
        (
            Jsonb([c.to_json() for c in columns]),
            description and description.text,
            description and str(description.source),
            str(status),
            name,
        ),
    )


def _carried_over(column: Column, old: Column | None) -> Column:
    """A re-uploaded column with the description and declared Join Hint its namesake had,
    if any. The hint is checked again once the new rows are stored."""
    if old is None:
        return column
    if old.description is not None:
        column = replace(column, description=old.description)
    if old.join_hint and old.join_hint.source == JoinHintSource.DECLARED:
        column = replace(column, join_hint=old.join_hint)
    return column


def list_datasets() -> list[Dataset]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT name, origin, columns, row_count, description, description_source,"
            " description_status FROM datasets ORDER BY name"
        ).fetchall()
    return [
        Dataset(
            r["name"],
            r["origin"],
            list(map(Column.from_json, r["columns"])),
            r["row_count"],
            Description.from_row(r),
            DescriptionStatus(r["description_status"]),
        )
        for r in rows
    ]


def run_query(
    query: str, *, row_limit: int = ROW_LIMIT, timeout_ms: int = STATEMENT_TIMEOUT_MS
) -> QueryResult:
    """Run one SQL statement as the read-only analyst role. Raises QueryError on failure.

    The limits hold whatever the SQL says: the statement runs prepared (the extended
    protocol, which accepts a single statement only) in a READ ONLY transaction whose
    timeout Perry sets first, and only `row_limit` rows are fetched.
    """
    timeout = sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(int(timeout_ms)))
    try:
        with psycopg.connect(db.analyst_conninfo(timeout_ms), connect_timeout=5) as conn:
            conn.read_only = True
            with conn.transaction():
                conn.execute(timeout)
                cursor = conn.execute(query, prepare=True)
                if cursor.description is None:
                    return QueryResult([], [], truncated=False)
                columns = [c.name for c in cursor.description]
                rows = cursor.fetchmany(row_limit + 1)
    except psycopg.errors.QueryCanceled as error:
        raise QueryError(
            f"The query took longer than {timeout_ms / 1000:g} seconds and was stopped."
        ) from error
    except psycopg.Error as error:
        message = error.diag.message_primary or str(error).strip()
        raise QueryError(f"The query failed: {message}") from error
    return QueryResult(columns, rows[:row_limit], truncated=len(rows) > row_limit)
