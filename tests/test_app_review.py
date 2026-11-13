"""The Review: an upload is read and chunked first, the Owner looks at each page's
Chunks, and only what they approve is indexed, exactly as they saw it."""

import pytest
from fakes import FakeVision, fake_provider
from logins import request_as

from perry import app, db, ingest, models

NOTES = "# Budget\n\nThe marketing budget for Q3 is 40,000 dollars."


@pytest.fixture
def notes(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text(NOTES, encoding="utf-8")
    return path


def stored_chunks(name: str) -> list[tuple[int, str]]:
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT c.page, c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.name = %s ORDER BY c.id",
            (name,),
        ).fetchall()
    return [(row["page"], row["text"]) for row in rows]


def stored_as(name: str) -> tuple[str, str] | None:
    """The Reader and Chunking Strategy the Document called `name` was stored with."""
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT reader, chunking FROM documents WHERE name = %s", (name,)
        ).fetchone()
    return (row["reader"], row["chunking"]) if row else None


def test_reading_for_review_indexes_nothing(perry, notes):
    read, status, _, panel, *_ = app.read_upload(request_as(perry.user), [str(notes)])

    assert [document.name for document in read] == ["notes.md"]
    assert panel["visible"] is True
    assert "approve" in status
    assert stored_chunks("notes.md") == []


def test_approving_indexes_exactly_the_chunks_reviewed(perry, notes):
    read, *_ = app.read_upload(request_as(perry.user), [str(notes)])

    left, status, panel, *_ = app.approve_review(request_as(perry.user), read)

    assert stored_chunks("notes.md") == [(c.page, c.text) for c in read[0].chunks]
    assert "notes.md: 1 chunks" in status
    assert (left, panel["visible"]) == ([], False)


def test_discarding_indexes_nothing(perry, notes):
    app.read_upload(request_as(perry.user), [str(notes)])

    left, status, panel, *_ = app.discard_review()

    assert (left, panel["visible"]) == ([], False)
    assert "nothing indexed" in status
    assert stored_chunks("notes.md") == []


def test_the_reader_and_chunking_strategy_picked_are_the_ones_stored(perry, notes):
    read, *_ = app.read_upload(request_as(perry.user), [str(notes)], "pymupdf", "fixed")
    app.approve_review(request_as(perry.user), read)

    assert stored_as("notes.md") == ("pymupdf", "fixed")


def test_an_unchanged_upload_is_not_read_again(perry, notes):
    ingest.ingest_file(notes)

    files, status, choice = app.propose_upload(request_as(perry.user), [str(notes)])

    assert (files, choice["visible"]) == ([], False)
    assert "Already stored" in status


def test_the_review_shows_every_page_and_flags_one_that_couldnt_be_read(perry, half_scanned_pdf):
    reader = FakeVision("# Budget\n\nQ3: 40,000.", RuntimeError("model down"), model="reader")

    with models.use(fake_provider(perry.model, page_reader=reader)):
        read, *_ = app.read_upload(request_as(perry.user), [str(half_scanned_pdf)], "vision")
    _, summary, picker, image, chunks = app.review_panel(read)

    assert [label for label, _ in picker["choices"]] == [
        "report.pdf · page 1 · 1 Chunks",
        "report.pdf · page 2 · 0 Chunks · not read",
    ]
    assert image.size[0] > 0  # page 1, rendered
    assert "Q3: 40,000." in chunks
    assert read[0].fingerprint in summary
    _, page_two = app.review_page(read, picker["choices"][1][1])
    assert "couldn't be read" in page_two
