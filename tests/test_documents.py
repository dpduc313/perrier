import pymupdf
import pytest

from perry import models
from perry.documents import CHUNK_TOKENS, citation, page_chunks, parse, split_text


def test_pdf_citations_point_at_a_page():
    assert citation("report.pdf", 4) == "report.pdf#page=4"
    assert citation("Report.PDF", 1) == "Report.PDF#page=1"


def test_non_paged_citations_are_the_file_name():
    assert citation("notes.md", 1) == "notes.md"


def test_short_text_is_one_chunk():
    assert split_text("one\n\ntwo", size=100) == ["one\n\ntwo"]


def test_chunks_respect_size_and_overlap():
    paragraphs = [f"paragraph {i} " + "x" * 80 for i in range(10)]
    pieces = split_text("\n\n".join(paragraphs), size=300, overlap=50)

    assert len(pieces) > 1
    assert all(len(p) <= 300 + 50 + 2 for p in pieces)
    # Every paragraph survives somewhere.
    for paragraph in paragraphs:
        assert any(paragraph in p for p in pieces)


def test_oversized_paragraph_is_hard_split():
    pieces = split_text("y" * 1000, size=300, overlap=50)
    assert all(len(p) <= 300 for p in pieces)
    assert "".join(p[:250] for p in pieces[:-1]) + pieces[-1] == "y" * 1000


def test_parse_markdown(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("# Title\n\nSome content.", encoding="utf-8")

    parsed = parse(file)

    assert parsed.name == "notes.md"
    assert [c.page for c in parsed.chunks] == [1]
    assert parsed.pages_without_text == []
    assert parsed.chunking == "heading"


HANDBOOK = (
    "# Handbook\n\n"
    "## Leave\n\nAnnual leave is 12 days a year.\n\n"
    "## Overtime\n\nOvertime on a weekday pays 150%.\n"
)


def test_heading_chunks_keep_to_one_section_and_lead_with_its_headings(tmp_path):
    file = tmp_path / "handbook.md"
    file.write_text(HANDBOOK, encoding="utf-8")

    texts = [chunk.text for chunk in parse(file, "heading").chunks]

    assert "Handbook\nLeave\nAnnual leave is 12 days a year." in texts
    assert "Handbook\nOvertime\nOvertime on a weekday pays 150%." in texts
    assert not any("Annual leave" in t and "Overtime on" in t for t in texts)


def test_fixed_chunks_ignore_sections(tmp_path):
    file = tmp_path / "handbook.md"
    file.write_text(HANDBOOK, encoding="utf-8")

    parsed = parse(file, "fixed")

    assert parsed.chunking == "fixed"
    assert len(parsed.chunks) == 1
    assert "Annual leave" in parsed.chunks[0].text
    assert "Overtime on" in parsed.chunks[0].text


def test_heading_chunks_stay_within_the_token_ceiling(tmp_path):
    file = tmp_path / "long.md"
    paragraphs = [f"Paragraph {i} " + "word " * 60 for i in range(40)]
    file.write_text("# Long\n\n" + "\n\n".join(paragraphs), encoding="utf-8")

    chunks = parse(file, "heading").chunks

    assert len(chunks) > 1
    assert all(models.count_embedding_tokens(c.text) <= CHUNK_TOKENS for c in chunks)
    for i in range(40):
        assert any(f"Paragraph {i} " in c.text for c in chunks)


def test_heading_chunks_never_span_two_pages(tmp_path):
    """Docling would merge two short passages under the same heading; a Chunk spanning
    pages would cite only the first, so each page keeps its own."""
    path = tmp_path / "report.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "The marketing budget for Q3 is 40,000 dollars.")
        pdf.new_page().insert_text((72, 72), "The hiring budget for Q3 is 25,000 dollars.")
        pdf.save(path)

    chunks = parse(path, "heading").chunks

    assert [(c.page, "40,000" in c.text, "25,000" in c.text) for c in chunks] == [
        (1, True, False),
        (2, False, True),
    ]


def test_an_unknown_chunking_is_refused(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("Some content.", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown chunking"):
        parse(file, "sentences")


def test_an_unknown_reader_is_refused(tmp_path):
    file = tmp_path / "notes.md"
    file.write_text("Some content.", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown reader"):
        parse(file, reader="ocr")


@pytest.fixture
def two_page_pdf(tmp_path):
    path = tmp_path / "plan.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "The marketing budget for Q3 is 40,000 dollars.")
        pdf.new_page().insert_text((72, 72), "The hiring budget for Q3 is 25,000 dollars.")
        pdf.save(path)
    return path


def test_the_pymupdf_reader_chunks_each_pages_text_layer(two_page_pdf):
    for chunking in ("heading", "fixed"):
        parsed = parse(two_page_pdf, chunking, "pymupdf")

        assert parsed.reader == "pymupdf"
        assert [(c.page, c.text) for c in parsed.chunks] == [
            (1, "The marketing budget for Q3 is 40,000 dollars."),
            (2, "The hiring budget for Q3 is 25,000 dollars."),
        ]
        assert parsed.pages_to_read == []


def test_the_vision_reader_leaves_every_pdf_page_to_the_model(two_page_pdf, tmp_path):
    parsed = parse(two_page_pdf, reader="vision")

    assert parsed.read_by_vision_reader
    assert parsed.chunks == []
    assert parsed.pages_to_read == [1, 2]

    notes = tmp_path / "notes.md"  # text files need no reading
    notes.write_text("Some content.", encoding="utf-8")
    parsed = parse(notes, reader="vision")
    assert not parsed.read_by_vision_reader
    assert [c.text for c in parsed.chunks] == ["Some content."]


def test_a_page_read_as_markdown_is_chunked_by_its_headings():
    markdown = "# Leave\n\nAnnual leave is 12 days.\n\n# Overtime\n\nWeekdays pay 150%."

    assert [(c.page, c.text) for c in page_chunks(3, markdown, "heading")] == [
        (3, "Leave\nAnnual leave is 12 days."),
        (3, "Overtime\nWeekdays pay 150%."),
    ]
    assert [c.page for c in page_chunks(3, markdown, "fixed")] == [3]
    assert page_chunks(3, "", "heading") == []
