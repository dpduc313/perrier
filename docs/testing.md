# Testing a change to Perry

How to check a new feature before it's merged, from the cheapest check to the most realistic. Most changes need steps 1 and 2. A change to the models, fallbacks, prompts or Retrieval Techniques also needs 3 and 4, and the PR should show what a Turn costs before and after.

| Step | What it checks | Cost | Time |
|---|---|---|---|
| 1. Automated tests | Behaviour, with fake models | free | about 13 min for the whole suite |
| 2. The feature's own tests | Just the part you changed | free | seconds |
| 3. Live questions with `scripts.ask` | Real models: answers, time, cost per Turn | free on Gemini and inet-qwen; paid only for OpenAI calls | about 30 s a Turn on Gemini's free tier |
| 4. In the app | What a User sees | as step 3 | minutes |
| 5. Eval set | Routing, Citations and answer quality over 48 Turns | about $0.30 on gpt-5.4-mini | about 15 min |

## Work in a worktree

Perry is served on a fixed port (7860), and a running demo reads `prices.toml` from its checkout, so try a branch in its own worktree:

```
git worktree add ../perry-my-change -b my-change main
copy .env ..\perry-my-change\.env
cd ..\perry-my-change
uv run pytest -q          # the first run creates the worktree's .venv (about a minute)
```

Each checkout gets its own `perry_test_*` database, so worktrees never share test data. When the branch is merged, remove the worktree with `git worktree remove ../perry-my-change`.

## 1. Automated tests

```
docker compose up -d
uv run pytest -q
uv run ruff check .
```

The git hooks run both on every commit and merge, and so does CI. The tests never call a real model: `tests/fakes.py` scripts every reply. That makes them free and repeatable, but it means they can't tell you whether a real model answers well.

## 2. The feature's own tests

Run only the files you touched while you work, for example:

```
uv run pytest tests/test_model_fallback.py tests/test_company_model.py -q
```

Write the test for each behaviour through the public interface, as a User would see it: `perry.send()` in `tests/conftest.py` runs a whole Turn, and `perry.script()` queues the fake models' replies. For a fallback, script the failure itself, e.g. `openai.APIConnectionError(...)` for an unreachable server. `tests/test_company_model.py` has examples.

## 3. Live questions with `scripts.ask`

`scripts.ask` asks real questions on your own Documents (the database in `.env`) and prints, for each Turn, the reply, how long it took, and every model call from the Usage Ledger with its tokens and cost:

```
uv run python -m scripts.ask "How is the Q3 marketing budget split?"
```

```
=== How is the Q3 marketing budget split?  (turn, 122.8s, $0.0000)
The Q3 marketing budget of $40,000 is split as follows: ...
Found with: Hybrid search, reranked (picked logically) · inet-qwen

  model                            kind           calls      in    out      cost
  gemini-3.5-flash-lite            chat               2    2501    177 $  0.0000
  knowledge                        chat               2    3051    157 $  0.0000
  nomic-embed-text                 embedding          1       7      0 $  0.0000

  stage                         seconds  share  detail
  Supervisor chooses Agents         1.7     2%  1 calls
  Agents run                       54.9    74%  1 asked: ask_knowledge_agent 54.9s
  Supervisor writes answer          1.7     2%  1 calls
  Other (Thread, checkpoints)      15.4    21%
  Turn                             73.7   100%
```

The stages split the Turn's time: the Supervisor's model calls before its last Agent, the Agents it asked (from the first one's start to the last one's end, since they can run at once), its model calls after them, and the rest. With several Turns, a table gives each Turn's stages and their medians.

The first Turn in a new checkout also loads the local reranker and ColBERT, which takes a minute or so; later Turns take about 10 s on Flash-Lite with inet-qwen.

- `--knowledge` asks the Knowledge Agent only, without the Supervisor.
- `--technique hyde` forces a Retrieval Technique.
- `--repeat 3` asks each question 3 times and prints the median time and the cost per Turn.
- `--label "..."` saves a note with each row, such as the setup you're trying.

A full Turn runs as the Owner on a Thread made for it and deleted afterwards. Its Usage Ledger rows are kept, so it counts toward today's Daily Cap like any Turn. A Turn that proposes a Write Action stops at the Confirmation, and nothing is performed.

### Recording the cost per Turn

Every Turn is appended to `scripts/results/asks.jsonl` (not committed), with the git branch and commit, the model settings it ran under, the time, the cost, and each model's calls. To compare two versions, ask the same questions on each and compare the rows:

```
cd ..\Perry
uv run python -m scripts.ask "Q1" "Q2" "Q3" --repeat 2 --label "main"
cd ..\perry-my-change
uv run python -m scripts.ask "Q1" "Q2" "Q3" --repeat 2 --label "my change"
```

Put the before and after in the PR's Evidence section: median time, cost per Turn, and which models answered.

### Forcing each fallback

Override a setting for one terminal session to make each part of the model chain answer. Each line shows what the "Found with" line should then end with:

| Setup | Override | "Found with" ends with |
|---|---|---|
| Normal | none | `· inet-qwen` |
| inet-qwen unreachable | `set PERRY_INET_QWEN_BASE_URL=http://127.0.0.1:9/v1` | `· Gemini (inet-qwen unavailable)` |
| Only paid OpenAI answers | `set PERRY_CHAT_TIMEOUT=0.01` | `· OpenAI (inet-qwen, Gemini unavailable)` (before PR #52: `· OpenAI (inet-qwen unavailable)`) |
| Paid chat for the whole Turn | `set PERRY_PAID_CHAT_ONLY=true` | `· inet-qwen` (the Supervisor runs on OpenAI) |

The last two make paid calls, about $0.01 a Turn. Close the terminal afterwards, or set the variables back to empty, so the overrides don't leak into the next run.

## 4. In the app

Stop any Perry that's running, then start the branch's:

```
uv run python -m perry.web
```

Ask the same questions in the chat. Check the "Found with" line and the sidebar's usage totals for today. Overrides go in `.env`, and need a restart. Afterwards, restart Perry from the checkout you demo from.

## 5. Eval set

Before switching models or rewriting a prompt, run the eval set on both versions. It gates on routing and Citation checks and reports a judge score:

```
set PERRY_BENCHMARK_LLM_MODEL=gpt-5.4-mini
uv run python -m evals --paid
```

It uses its own database (`perry_eval`), so it never touches your Documents. Estimate the cost before running: the last full runs used about 5.6k input and 0.45k output tokens per Turn, about $0.006 a Turn on gpt-5.4-mini. See the README's "Eval set" section for the details.
