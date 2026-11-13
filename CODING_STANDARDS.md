# Coding standards

Read at review time. Ruff (`uv run ruff check .`, `uv run ruff format --check .`) and the tests already enforce the mechanical rules, including that only `perry/models.py` imports a model provider or reads its settings. These are the judgement calls left over.

## Vocabulary

Names in code, prompts, docs and messages use the terms in `GLOSSARY.md` (Supervisor, Agent, Turn, Thread, Document, Chunk, Citation, Write Action, Confirmation, …), with the meaning given there. A new domain concept gets a glossary entry in the same change.

## Decisions

Code follows the accepted ADRs in `docs/adr/`. A change that departs from one also updates or supersedes that ADR in the same PR: for example, a gate counting checks the ADR didn't name.

## Knowledge Agent interface

The Knowledge Agent stays retrieval-agnostic: a question goes in, cited findings come out (`docs/adr/0001`). Retrieval details (ColBERT, RAPTOR, reranking, techniques) stay behind that interface, and no caller branches on them.
