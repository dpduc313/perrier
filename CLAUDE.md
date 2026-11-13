# Perry

- **Vocabulary**: `GLOSSARY.md`. Review rules: `CODING_STANDARDS.md`.
- **Model provider**: `perry/models.py` is the only module that knows the model providers: Gemini free tier for chat, a paid OpenAI fallback when its quota runs out, the company's `inet-qwen` (vLLM, chat only, no tool calls, so tools become JSON choices) first for knowledge work when `PERRY_INET_QWEN_BASE_URL` is set, and for whatever else `PERRY_INET_QWEN_FOR` names (`docs/adr/0006`, `0007`), local Ollama for embeddings and Summaries, and ColBERT (`docs/adr/0003`). Switching a model is a change there only.
- **Database**: one Postgres (`compose.yaml`, port 5433). The one exception is ColBERT's on-disk index, which ingestion keeps in step with Postgres (`docs/adr/0002`). Connect via `127.0.0.1`, not `localhost`: on Windows, `localhost` tries IPv6 first and stalls. The pool prepares every statement (the LangGraph checkpointer requires it), so multi-statement SQL needs `prepare=False`.
- **Library APIs**: langchain 1.x / langgraph 1.x / gradio 6.x differ a lot from older tutorials. Verify signatures against `.venv/Lib/site-packages`. Gradio 6 findings: `docs/gradio.md`.
- **Trying a change**: `docs/testing.md`: worktrees, live questions with `uv run python -m scripts.ask` (records each Turn's time and cost per model), forcing each fallback, and the eval set. Put a model or prompt change's before/after cost per Turn in the PR.
- **Checks**: `uv run pytest` (DB tests need `docker compose up -d`; each checkout gets its own `perry_test_*` database) and `uv run ruff check .`. The git hooks and CI run both; a commit or merge waits 5 to 15 minutes for the full suite, so give it a long timeout.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `dpduc/Perry` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
