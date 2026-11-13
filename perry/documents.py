"""Turn a Document file into page-numbered Chunks, so every answer can cite a page.

A Reader turns a PDF into text, page by page (`PERRY_DOCUMENT_READER`):

- "docling" (the default): Docling's layout model finds the headings, lists and tables.
- "pymupdf": the text layer as PyMuPDF extracts it, with no structure.
- "vision": a vision model (`PERRY_PAGE_READER_MODEL`, gpt-5.4-mini) reads each page,
  rendered as an image, into Markdown. It calls a paid model, so perry.ingest does the
  reading; this module only lists the pages and chunks the Markdown that comes back.

Markdown and text files read the same whichever Reader is chosen.

Then a Chunking Strategy splits the text (`PERRY_CHUNKING`):

- "heading" (the default): each Chunk stays within one section and one page, led by its
  section's headings, and holds at most CHUNK_TOKENS of the embedding model's tokens.
- "fixed": each page's text is cut into pieces of about CHUNK_CHARS characters,
  overlapping by OVERLAP_CHARS, whatever the sections.

Every combination exists so they can be compared (docs/adr/0009):
`uv run python -m benchmark.run perry-retrieval --reader pymupdf --chunking fixed`.
"""

import hashlib
import io
import os
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import pymupdf
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types import DoclingDocument

from perry import models

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}

# Resolution pages are rendered at for a vision model: readable small print, without
# paying for more image tokens than the text needs.
RENDER_DPI = 150


def _setting(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
    return value


READERS = ("docling", "pymupdf", "vision")
READER = _setting("PERRY_DOCUMENT_READER", "docling", READERS)
CHUNKINGS = ("heading", "fixed")
CHUNKING = _setting("PERRY_CHUNKING", "heading", CHUNKINGS)

# Fixed chunking: roughly 300-400 tokens, small enough to retrieve precisely, big enough
# to keep context. Heading chunking's ceiling is about the same size, so the two compare.
CHUNK_CHARS = 1500
OVERLAP_CHARS = 200
CHUNK_TOKENS = 384


@dataclass(frozen=True)
class Chunk:
    page: int  # 1-based; non-paged files are page 1
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    name: str
    content_hash: str
    chunks: list[Chunk]
    # PDF pages with no text layer (e.g. scanned pages): render them to transcribe them.
    pages_without_text: list[int]
    reader: str  # the Reader the text came from
    chunking: str  # the Chunking Strategy that made `chunks`
    # The pages a vision model still has to read, chunked with `page_chunks`: every page
    # for the vision Reader (`chunks` is then empty), else the pages without text.
    pages_to_read: list[int]

    @property
    def read_by_vision_reader(self) -> bool:
        """Whether the vision Reader reads every page: a PDF, with the "vision" Reader."""
        return self.reader == "vision" and self.name.lower().endswith(".pdf")


def citation(document_name: str, page: int) -> str:
    if document_name.lower().endswith(".pdf"):
        return f"{document_name}#page={page}"
    return document_name


def content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path, chunking: str | None = None, reader: str | None = None) -> ParsedDocument:
    """`path`'s Chunks, read with `reader` and split with `chunking` (PERRY_DOCUMENT_READER
    and PERRY_CHUNKING if None). Reading a PDF with Docling takes seconds, so the last few
    results are kept: estimating an upload's cost and then ingesting it read it once."""
    chunking = chunking or CHUNKING
    reader = reader or READER
    if chunking not in CHUNKINGS:
        raise ValueError(f"Unknown chunking {chunking!r}: use one of {', '.join(CHUNKINGS)}")
    if reader not in READERS:
        raise ValueError(f"Unknown reader {reader!r}: use one of {', '.join(READERS)}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name}")
    return _parse(path.resolve(), content_hash(path), chunking, reader)


def forget_parsed() -> None:
    """Forget the kept results, so the next `parse` reads its file again (the benchmark
    times reading). Docling's models stay loaded."""
    _parse.cache_clear()
    _convert.cache_clear()


@lru_cache(maxsize=16)
def _parse(path: Path, digest: str, chunking: str, reader: str) -> ParsedDocument:
    is_pdf = path.suffix.lower() == ".pdf"
    without_text = _pages_without_text(path)
    if is_pdf and reader == "vision":
        chunks: list[Chunk] = []
        to_read = list(range(1, page_count(path) + 1))
    else:
        if not is_pdf or reader == "docling":
            chunks = _chunks(_convert(path, digest), chunking)
        else:
            chunks = [
                c
                for number, text in _pymupdf_pages(path)
                for c in page_chunks(number, text, chunking)
            ]
        # A page without a text layer is read by the vision model instead (perry.ingest).
        chunks = [chunk for chunk in chunks if chunk.page not in without_text]
        to_read = without_text if is_pdf else []
    return ParsedDocument(
        name=path.name,
        content_hash=digest,
        chunks=chunks,
        pages_without_text=without_text,
        reader=reader,
        chunking=chunking,
        pages_to_read=to_read,
    )


def page_chunks(page: int, text: str, chunking: str | None = None) -> list[Chunk]:
    """The Chunks of one page's text (plain, or Markdown as a vision model reads it)."""
    if (chunking or CHUNKING) == "fixed":
        return [Chunk(page=page, text=piece) for piece in split_text(text)]
    if not text.strip():
        return []
    return _heading_chunks(_markdown(text), page=page)


def _chunks(document: DoclingDocument, chunking: str) -> list[Chunk]:
    if chunking == "heading":
        return _heading_chunks(document)
    return [
        c
        for page in PageChunker().chunk(document)
        for c in page_chunks(_page(page), page.text, "fixed")
    ]


def _pages_without_text(path: Path) -> list[int]:
    """The PDF pages with no text layer (e.g. scanned pages), 1-based."""
    if path.suffix.lower() != ".pdf":
        return [] if path.read_text(encoding="utf-8").strip() else [1]
    return [number for number, text in _pymupdf_pages(path) if not text.strip()]


def _pymupdf_pages(path: Path) -> list[tuple[int, str]]:
    with pymupdf.open(path) as pdf:
        return [(number, page.get_text()) for number, page in enumerate(pdf, start=1)]


def page_count(path: Path) -> int:
    """How many pages the file has: a PDF's page count; 1 for Markdown and text."""
    if path.suffix.lower() != ".pdf":
        return 1
    with pymupdf.open(path) as pdf:
        return pdf.page_count


@lru_cache(maxsize=8)
def _convert(path: Path, digest: str) -> DoclingDocument:
    """The file as Docling reads it; `digest` (its content hash) keys the cache."""
    if path.suffix.lower() == ".pdf":
        return _converter().convert(path).document
    # Docling has no plain-text reader; plain text reads as Markdown paragraphs.
    return _markdown(path.read_text(encoding="utf-8"), name=path.stem)


def _markdown(text: str, name: str = "page") -> DoclingDocument:
    stream = DocumentStream(name=f"{name}.md", stream=io.BytesIO(text.encode("utf-8")))
    return _converter().convert(stream).document


@cache
def _converter() -> DocumentConverter:
    # No OCR: pages without a text layer go to the vision model, as before Docling.
    # pypdfium2, not Docling's default PDF reader: on a page that is only an image, the
    # default took two minutes (and crashed natively inside) where this takes seconds,
    # with the same text out.
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.MD],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=False),
                backend=PyPdfiumDocumentBackend,
            )
        },
    )


def _page(docling_chunk) -> int:
    """The first page a Docling chunk's items come from (1 for files without pages)."""
    pages = [prov.page_no for item in docling_chunk.meta.doc_items for prov in item.prov]
    return min(pages, default=1)


class _EmbeddingTokens(BaseTokenizer):
    """Docling counts tokens the way the embedding model does."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        return models.count_embedding_tokens(text)

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return None


def _heading_chunks(
    document: DoclingDocument, page: int | None = None, max_tokens: int = CHUNK_TOKENS
) -> list[Chunk]:
    """One Chunk per run of neighbouring passages under the same headings on the same page,
    up to `max_tokens`. Docling's own merging ignores pages, so a merged Chunk could span
    two and its Citation name only one; this merges within a page instead. `page` is the
    page of a document that is one page's text."""
    chunker = HybridChunker(tokenizer=_EmbeddingTokens(max_tokens=max_tokens), merge_peers=False)
    chunks: list[Chunk] = []
    key: tuple[int, tuple[str, ...]] | None = None
    texts: list[str] = []

    def flush() -> None:
        if texts:
            chunks.append(Chunk(page=key[0], text=_with_headings(key[1], "\n".join(texts))))

    for piece in chunker.chunk(document):
        piece_key = (page or _page(piece), tuple(piece.meta.headings or ()))
        merged = _with_headings(piece_key[1], "\n".join([*texts, piece.text]))
        if piece_key != key or models.count_embedding_tokens(merged) > max_tokens:
            flush()
            key, texts = piece_key, []
        texts.append(piece.text)
    flush()
    return chunks


def _with_headings(headings: tuple[str, ...], text: str) -> str:
    """The text as embedded and shown: its section's headings first, as Docling does."""
    return "\n".join([*headings, text])


def render_page(path: Path, page: int) -> bytes:
    """Page `page` (1-based) of the PDF at `path`, as a PNG image."""
    with pymupdf.open(path) as pdf:
        return pdf[page - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split on paragraph boundaries where possible, hard-splitting oversized paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > size:
            head, paragraph = paragraph[:size], paragraph[size - overlap :]
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if current and len(current) + 2 + len(paragraph) > size:
            pieces.append(current)
            # Carry the tail forward so a fact split across the boundary stays findable.
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def find_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
