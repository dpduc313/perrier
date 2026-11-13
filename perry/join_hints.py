"""Join Hints between Datasets: inferring them from column names, checking them against
the data whenever either Dataset is uploaded, and keeping a declared one that points at a
Dataset not uploaded yet until it arrives (ADR-0008).

A declared Join Hint comes from a Data Dictionary (perry.data_dictionary) and wins over an
inferred one for the same column. A hint guides the Analysis Agent's SQL; it never
refuses data.
"""

from dataclasses import dataclass, replace

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from perry.datasets import SCHEMA, Column, JoinHint, JoinHintSource


@dataclass(frozen=True)
class CheckedJoinHints:
    columns: list[Column]  # the uploaded Dataset's, with their Join Hints as checked
    dropped: list[str]  # declared hints the data no longer supports, any Dataset's
    orphaned: list[str]  # declared hints kept with values not found in their target
    waiting: list[str]  # declared hints whose target Dataset isn't uploaded yet


def recheck(conn: psycopg.Connection, uploaded: str) -> CheckedJoinHints:
    """Check the Join Hints both ways between the uploaded Dataset and the others, store
    the ones that changed, and return the uploaded Dataset's columns with what the upload
    reply says of the declared hints.

    A declared hint keeps its orphan count up to date, waits while its target Dataset
    isn't uploaded, and is dropped when its target column is gone or none of its values
    appear there. A column named `x_id` without one references another Dataset whose first
    column is `x_id` when every non-empty value of the column appears there.
    """
    registry = {
        r["name"]: list(map(Column.from_json, r["columns"]))
        for r in conn.execute("SELECT name, columns FROM datasets ORDER BY name").fetchall()
    }
    names = {name: [c.name for c in columns] for name, columns in registry.items()}
    keys = {name: columns[0].name for name, columns in registry.items() if columns}
    dropped, orphaned, waiting = [], [], []
    for name, columns in registry.items():
        checked = []
        for column in columns:
            hint = column.join_hint
            if hint and hint.source == JoinHintSource.DECLARED and uploaded in (name, hint.dataset):
                if target_not_uploaded(hint, names):  # from a row that waited for `uploaded`
                    wait_for_target(conn, name, column.name, hint)
                    waiting.append(
                        f"{name}.{column.name} → {hint.target()} ({hint.dataset} isn't uploaded yet)"
                    )
                    found = None
                else:
                    found = check_declared(conn, names, name, column.name, hint)
                if isinstance(found, str):
                    dropped.append(f"{name}.{column.name} → {hint.target()} ({found})")
                    found = None
                column = replace(column, join_hint=found)
                if found and found.orphans:
                    orphaned.append(describe_orphans(name, column))
            if _may_change_inferred_hint(name, column, uploaded, keys):
                column = _with_inferred_hint(conn, name, column, keys)
            checked.append(column)
        if checked != columns:
            _save_columns(conn, name, checked)
            registry[name] = checked
    return CheckedJoinHints(registry[uploaded], dropped, orphaned, waiting)


def _save_columns(conn: psycopg.Connection, name: str, columns: list[Column]) -> None:
    conn.execute(
        "UPDATE datasets SET columns = %s WHERE name = %s",
        (Jsonb([c.to_json() for c in columns]), name),
    )


def _may_change_inferred_hint(name: str, column: Column, uploaded: str, keys: dict) -> bool:
    """Whether the upload may have made or broken the column's inferred Join Hint."""
    hint = column.join_hint
    if hint and hint.source != JoinHintSource.INFERRED:
        return False  # a declared hint wins
    if name == uploaded:
        return column.name.endswith("_id")
    if hint:
        return hint.dataset == uploaded  # its target was replaced
    return keys.get(uploaded) == column.name  # the upload may be its target


def _with_inferred_hint(conn: psycopg.Connection, name: str, column: Column, keys: dict) -> Column:
    """The column with the Join Hint its name and values support now, or none."""
    for target in sorted(t for t, key in keys.items() if key == column.name and t != name):
        if _all_found(conn, name, target, column.name):
            return replace(column, join_hint=JoinHint(target, column.name))
    return replace(column, join_hint=None)


def _all_found(conn: psycopg.Connection, source: str, target: str, column: str) -> bool:
    """Whether the source column has values and every one appears in the target column.

    Compared as text, so 3 in a bigint key matches 3.0 in a column that blanks made
    double precision.
    """
    query = sql.SQL(
        """
        SELECT EXISTS (SELECT 1 FROM {source} WHERE {column} IS NOT NULL)
           AND NOT EXISTS (
               SELECT 1 FROM {source} s WHERE s.{column} IS NOT NULL AND NOT EXISTS (
                   SELECT 1 FROM {target} t WHERE t.{column}::text = s.{column}::text))
           AS found
        """
    ).format(
        source=sql.Identifier(SCHEMA, source),
        target=sql.Identifier(SCHEMA, target),
        column=sql.Identifier(column),
    )
    return conn.execute(query).fetchone()["found"]


def check_declared(
    conn: psycopg.Connection,
    names: dict[str, list[str]],
    dataset: str,
    column: str,
    hint: JoinHint,
) -> JoinHint | str:
    """The declared hint with its orphan count, or why it is refused. `names` maps each
    Dataset to its column names, and must hold the hint's target Dataset."""
    if not hint.dataset:
        return f"{hint.column} isn't written as dataset.column"
    if hint.column not in names[hint.dataset]:
        return f"{hint.dataset} has no column {hint.column}"
    values, orphans = _count_orphans(conn, dataset, column, hint)
    if orphans == values:
        return f"none of {dataset}.{column}'s values appear in {hint.target()}"
    return replace(hint, orphans=orphans)


def _count_orphans(
    conn: psycopg.Connection, dataset: str, column: str, hint: JoinHint
) -> tuple[int, int]:
    """How many distinct non-empty values the column has, and how many of them don't
    appear in the hint's target column. Compared as text, as _all_found does."""
    query = sql.SQL(
        """
        SELECT count(*) AS values, count(*) FILTER (WHERE NOT EXISTS (
                   SELECT 1 FROM {target} t WHERE t.{key}::text = s.value)) AS orphans
        FROM (SELECT DISTINCT {column}::text AS value FROM {source}
              WHERE {column} IS NOT NULL) s
        """
    ).format(
        source=sql.Identifier(SCHEMA, dataset),
        column=sql.Identifier(column),
        target=sql.Identifier(SCHEMA, hint.dataset),
        key=sql.Identifier(hint.column),
    )
    found = conn.execute(query).fetchone()
    return found["values"], found["orphans"]


def describe_orphans(dataset: str, column: Column) -> str:
    """What the upload reply says of a kept declared hint with values not in its target."""
    hint = column.join_hint
    return f"{_values(hint.orphans)} of {dataset}.{column.name} not found in {hint.target()}"


def _values(n: int) -> str:
    return f"{n} value{'s' if n != 1 else ''}"


# Declared Join Hints waiting for the Dataset they point at.


def target_not_uploaded(hint: JoinHint, names: dict[str, list[str]]) -> bool:
    return bool(hint.dataset) and hint.dataset not in names


def wait_for_target(conn: psycopg.Connection, dataset: str, column: str, hint: JoinHint) -> None:
    """Keep a declared hint until the Dataset it points at is uploaded."""
    conn.execute(
        "INSERT INTO waiting_join_hints (dataset, column_name, target_dataset, target_column)"
        " VALUES (%s, %s, %s, %s) ON CONFLICT (dataset, column_name) DO UPDATE"
        " SET target_dataset = excluded.target_dataset, target_column = excluded.target_column",
        (dataset, column, hint.dataset, hint.column),
    )


def forget_waiting_except(conn: psycopg.Connection, dataset: str, columns: list[str]) -> None:
    """Forget the Dataset's declared hints waiting for their target, but those of
    `columns`."""
    conn.execute(
        "DELETE FROM waiting_join_hints WHERE dataset = %s AND NOT column_name = ANY(%s)",
        (dataset, columns),
    )


def attach_waiting(conn: psycopg.Connection, target: str) -> None:
    """Attach the declared hints that waited for `target`, unchecked: `recheck` after the
    upload checks them."""
    waiting: dict[str, dict[str, str]] = {}
    for r in conn.execute(
        "DELETE FROM waiting_join_hints WHERE target_dataset = %s"
        " RETURNING dataset, column_name, target_column",
        (target,),
    ).fetchall():
        waiting.setdefault(r["dataset"], {})[r["column_name"]] = r["target_column"]
    for dataset, hints in sorted(waiting.items()):
        row = conn.execute(
            "SELECT columns FROM datasets WHERE name = %s FOR UPDATE", (dataset,)
        ).fetchone()
        columns = [
            replace(c, join_hint=JoinHint(target, hints[c.name], JoinHintSource.DECLARED))
            if c.name in hints
            else c
            for c in map(Column.from_json, row["columns"])
        ]
        _save_columns(conn, dataset, columns)
