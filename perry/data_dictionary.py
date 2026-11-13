"""Data Dictionaries: files that describe Datasets instead of being one (ADR-0008).

`datasets.upload()` hands a `.dictionary.csv` file, or a workbook's `dictionary` sheet,
to `parse` and `apply`. Rows for a Dataset not uploaded yet wait in their own table until
its upload claims them (`attach_waiting_rows`). A declared Join Hint is checked against
the data by perry.join_hints, and waits there while its target isn't uploaded.
"""

from dataclasses import dataclass, field, replace

import pandas as pd
import psycopg

from perry import db, join_hints
from perry.datasets import (
    ITSELF,
    Column,
    DatasetItself,
    Description,
    DescriptionSource,
    JoinHint,
    JoinHintSource,
    count_described,
    description_status,
    identifier,
    resolve_description,
    save_descriptions,
)

SUFFIX = ".dictionary"  # payroll.dictionary.csv describes payroll
SHEET = "dictionary"  # a workbook sheet that describes the other sheets


class DictionaryRefused(ValueError):
    """A Data Dictionary refused whole, with a message saying why."""


@dataclass(frozen=True)
class RefusedDictionary:
    """A workbook's `dictionary` sheet refused whole, beside the sheets it couldn't describe."""

    origin: str
    why: str  # DictionaryRefused's message

    def describe(self) -> str:
        return f"{self.origin}: {self.why}"


@dataclass(frozen=True)
class DictionaryRow:
    line: int  # where it is in the file, counting the header as line 1
    dataset: str  # normalised like a Dataset name
    column: str | DatasetItself  # normalised like a column name, or ITSELF
    description: str
    join_hint: JoinHint | None  # declared in `references`, not yet checked against the data

    @property
    def target(self) -> str:
        return self.dataset if self.column is ITSELF else f"{self.dataset}.{self.column}"


@dataclass(frozen=True)
class ParsedDictionary:
    rows: list[DictionaryRow]
    unmatched: list[tuple[int, str]]  # (line, why) for rows that can't attach anywhere


@dataclass(frozen=True)
class DictionaryResult:
    origin: str
    described: dict[str, tuple[bool, int]]  # Dataset -> (its own description?, columns)
    unmatched: list[tuple[int, str]]  # (line, why), in file order
    # Datasets not uploaded yet -> (their own description?, columns): kept until they are
    waiting: dict[str, tuple[bool, int]] = field(default_factory=dict)
    refused: list[tuple[int, str]] = field(default_factory=list)  # declared Join Hints
    orphaned: list[tuple[int, str]] = field(default_factory=list)  # kept, values not found
    # declared Join Hints whose target Dataset isn't uploaded yet: attached when it is
    join_hints_waiting: list[tuple[int, str]] = field(default_factory=list)

    def describe(self) -> str:
        parts = [_described(name, *counts) for name, counts in self.described.items()]
        line = f"{self.origin}: Data Dictionary, described {', '.join(parts) or 'nothing'}"
        if self.waiting:
            parts = [_described(name, *counts) for name, counts in self.waiting.items()]
            line += f"; waiting for {', '.join(parts)} to be uploaded"
        # "references", as the Owner wrote them in the dictionary's `references` column
        for label, rows in (
            ("not attached", self.unmatched),
            ("references refused", self.refused),
            ("values not found", self.orphaned),
            ("references waiting", self.join_hints_waiting),
        ):
            if rows:
                line += f"; {label}: " + ", ".join(f"row {n} ({why})" for n, why in rows)
        return line


def _described(name: str, itself: bool, columns: int) -> str:
    return f"{name} ({count_described(itself, columns)})"


def _text(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def parse(
    frame: pd.DataFrame, *, default: str | None = None, sheets: dict[str, str] | None = None
) -> ParsedDictionary:
    """The rows of a Data Dictionary, with names normalised as uploads normalise them.

    A blank `dataset` means `default`; in a workbook, a sheet's name means that sheet's
    Dataset (`sheets`). Raises DictionaryRefused when the file can't be applied at all.
    """
    frame = frame.rename(columns=lambda label: identifier(str(label), "column"))
    # Without a file name to stand for it, every row has to name its Dataset.
    needed = ["description"] if default else ["dataset", "description"]
    missing = next((c for c in needed if c not in frame.columns), None)
    if missing:
        raise DictionaryRefused(
            f"Data Dictionary refused: it has no {missing} column "
            "(its columns are dataset, column, description and references)"
        )
    rows, unmatched, seen = [], [], {}
    for line, record in enumerate(frame.to_dict("records"), start=2):
        dataset, column, description, references = (
            _text(record.get(key)) for key in ("dataset", "column", "description", "references")
        )
        if not (dataset or column or description or references):
            continue  # a blank line
        name = identifier(dataset, "dataset") if dataset else default
        if name is None:
            unmatched.append((line, "it names no Dataset"))
            continue
        name = (sheets or {}).get(name, name)
        row = DictionaryRow(
            line,
            name,
            identifier(column, "column") if column else ITSELF,
            description,
            parse_join_hint(references, sheets) if references else None,
        )
        if row.target in seen:  # even when one copy is blank: the Owner meant one of them
            raise DictionaryRefused(
                f"Data Dictionary refused: rows {seen[row.target]} and {line} "
                f"both describe {row.target}"
            )
        seen[row.target] = line
        if not description:
            unmatched.append((line, f"{row.target} has no description"))
            continue
        rows.append(row)
    return ParsedDictionary(rows, unmatched)


def parse_join_hint(text: str, sheets: dict[str, str] | None) -> JoinHint:
    """The Join Hint a Data Dictionary's `references` cell (`dataset.column`) declares."""
    dataset, _, column = text.rpartition(".")
    name = identifier(dataset, "dataset") if dataset else ""
    return JoinHint(
        (sheets or {}).get(name, name), identifier(column, "column"), JoinHintSource.DECLARED
    )


def apply(origin: str, parsed: ParsedDictionary) -> DictionaryResult:
    """Replace the Owner's descriptions and declared Join Hints of each Dataset the
    dictionary names, in one transaction. Datasets it doesn't name are left alone, and
    rows for a Dataset not uploaded yet wait for it. A declared hint waits while its target
    Dataset isn't uploaded; otherwise it is checked against the data: refused when its
    target column is missing or none of its values appear there, kept with its orphan
    count otherwise."""
    named: dict[str, list[DictionaryRow]] = {}
    for row in parsed.rows:
        named.setdefault(row.dataset, []).append(row)
    described, waiting, unmatched = {}, {}, list(parsed.unmatched)
    refused, orphaned, hints_waiting = [], [], []
    with db.pool().connection() as conn, conn.transaction():
        names = {
            r["name"]: [c["name"] for c in r["columns"]]
            for r in conn.execute("SELECT name, columns FROM datasets").fetchall()
        }
        found = {
            r["name"]: r
            for r in conn.execute(
                "SELECT name, columns, description, description_source FROM datasets"
                " WHERE name = ANY(%s) ORDER BY name FOR UPDATE",
                (list(named),),
            ).fetchall()
        }
        for name, rows in named.items():
            itself = any(r.column is ITSELF for r in rows)
            if name not in found:
                _keep_waiting(conn, origin, name, rows)
                waiting[name] = (itself, sum(r.column is not ITSELF for r in rows))
                continue
            columns = [Column.from_json(c) for c in found[name]["columns"]]
            given = {r.column: r for r in rows}
            unmatched += [
                (r.line, f"{name} has no column {r.column}")
                for r in rows
                if r.column is not ITSELF and r.column not in {c.name for c in columns}
            ]
            # the dictionary replaces what was declared for this Dataset, waiting or not
            join_hints.forget_waiting_except(conn, name, [])
            declared = []
            for column in columns:
                row = given.get(column.name)
                hint = row.join_hint if row else None
                if hint and join_hints.target_not_uploaded(hint, names):
                    join_hints.wait_for_target(conn, name, column.name, hint)
                    hints_waiting.append((row.line, f"{hint.dataset} isn't uploaded yet"))
                    hint = None
                column, why = _with_checked_join_hint(
                    conn, names, name, _with_owners_description(column, row), hint
                )
                if why:
                    refused.append((row.line, why))
                elif hint and column.join_hint.orphans:
                    orphaned.append((row.line, join_hints.describe_orphans(name, column)))
                declared.append(column)
            description = _owners_description(Description.from_row(found[name]), given.get(ITSELF))
            save_descriptions(
                conn, name, declared, description, description_status(description, declared)
            )
            described[name] = (itself, sum(c.name in given for c in declared))
            join_hints.recheck(conn, name)  # infers hints for columns that lost declared ones
    return DictionaryResult(
        origin,
        dict(sorted(described.items())),
        sorted(unmatched),
        dict(sorted(waiting.items())),
        sorted(refused),
        sorted(orphaned),
        sorted(hints_waiting),
    )


def _owners_description(
    current: Description | None, row: "DictionaryRow | WaitingRow | None"
) -> Description | None:
    """What a Dataset or column is described as after a Data Dictionary that names its
    Dataset, by the one rule for who describes what (resolve_description)."""
    given = Description(row.description, DescriptionSource.USER) if row else None
    return resolve_description(current, given)


def _with_owners_description(column: Column, row: "DictionaryRow | WaitingRow | None") -> Column:
    return replace(column, description=_owners_description(column.description, row))


def _with_declared_join_hint(column: Column, hint: JoinHint | None) -> Column:
    """The column with the Join Hint a Data Dictionary row declares, not yet checked. A
    column the dictionary declares nothing for loses its declared hint."""
    if hint:
        return replace(column, join_hint=hint)
    if column.join_hint and column.join_hint.source == JoinHintSource.DECLARED:
        return replace(column, join_hint=None)
    return column


def _with_checked_join_hint(
    conn: psycopg.Connection,
    names: dict[str, list[str]],
    dataset: str,
    column: Column,
    hint: JoinHint | None,
) -> tuple[Column, str | None]:
    """The column with the Join Hint its Data Dictionary row declares, checked against the
    data, and why that hint was refused, if it was. A column the dictionary declares
    nothing for loses its declared hint."""
    if hint is None:
        return _with_declared_join_hint(column, None), None
    checked = join_hints.check_declared(conn, names, dataset, column.name, hint)
    if isinstance(checked, str):
        return _with_declared_join_hint(column, None), checked
    return replace(column, join_hint=checked), None


# Rows waiting for their Dataset to be uploaded.


@dataclass(frozen=True)
class WaitingRow:
    """A Data Dictionary row kept for a Dataset until it was uploaded."""

    origin: str
    line: int
    column: str | DatasetItself
    description: str
    join_hint: JoinHint | None  # declared, checked once the Dataset's rows are stored


def _keep_waiting(
    conn: psycopg.Connection, origin: str, name: str, rows: list[DictionaryRow]
) -> None:
    """Keep a Dataset's rows until it is uploaded, replacing any kept before."""
    conn.execute("DELETE FROM waiting_descriptions WHERE dataset = %s", (name,))
    with conn.cursor() as cursor:
        cursor.executemany(
            "INSERT INTO waiting_descriptions"
            " (dataset, column_name, description, refers_to, origin, line)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (
                    name,
                    "" if r.column is ITSELF else r.column,
                    r.description,
                    r.join_hint and r.join_hint.target(),
                    origin,
                    r.line,
                )
                for r in rows
            ],
        )


def _claim_waiting(conn: psycopg.Connection, name: str) -> list[WaitingRow]:
    """The rows waiting for this Dataset, removed so they attach only once."""
    rows = conn.execute(
        "DELETE FROM waiting_descriptions WHERE dataset = %s"
        " RETURNING origin, line, column_name, description, refers_to",
        (name,),
    ).fetchall()
    waiting = [
        WaitingRow(
            r["origin"],
            r["line"],
            r["column_name"] or ITSELF,
            r["description"],
            parse_join_hint(r["refers_to"], None) if r["refers_to"] else None,
        )
        for r in rows
    ]
    return sorted(waiting, key=lambda w: w.line)


@dataclass(frozen=True)
class AttachedRows:
    columns: list[Column]  # described and with declared Join Hints, not yet checked
    description: Description | None
    not_attached: list[str]  # rows naming a column the Dataset lacks


def attach_waiting_rows(
    conn: psycopg.Connection, name: str, columns: list[Column], description: Description | None
) -> AttachedRows | None:
    """Apply the rows that waited for this Dataset to its columns and description, as the
    Data Dictionary they came from would have: None if no rows waited."""
    waiting = _claim_waiting(conn, name)
    if not waiting:
        return None
    given = {r.column: r for r in waiting}
    not_attached = [
        f"{r.origin} row {r.line} ({name} has no column {r.column})"
        for r in waiting
        if r.column is not ITSELF and r.column not in {c.name for c in columns}
    ]
    columns = [
        _with_declared_join_hint(
            _with_owners_description(c, given.get(c.name)),
            given[c.name].join_hint if c.name in given else None,
        )
        for c in columns
    ]
    return AttachedRows(columns, _owners_description(description, given.get(ITSELF)), not_attached)
