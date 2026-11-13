# Perry

A personal multi-agent assistant that runs on your own machine. You talk to one chat; a **Supervisor** hands each request to the right specialist **Agent** and answers with the results. Vocabulary: [GLOSSARY.md](GLOSSARY.md). Decisions: [docs/adr/](docs/adr/).

![Perry's chat UI: Threads, Documents and Datasets in the sidebar, today's usage, the Google account section, and the chat with a push-to-talk recorder below it](docs/images/perry-ui.png)

## What Perry can do

| Agent | What it does for you |
|---|---|
| **Knowledge Agent** | Answers questions from the Documents you upload (PDF, Markdown, text) and cites the exact page each claim came from. Picks one of ten Retrieval Techniques per question and says which one it used: hybrid search (pgvector + Postgres full-text, the baseline); Multi-query and RAG fusion, which rewrite the question into several hybrid searches; Question Decomposition, which splits a comparing or chaining question into sub-questions; Step-back, which also searches a broader version of "why" questions; HyDE, which searches with a hypothetical answer; Self-query, which limits the search to the Documents the question names (by name, file type, upload date, page, Document date or topic); Document Summaries and RAPTOR (summaries of related Chunks up to themes across Documents), both written in the background by a local model; and ColBERT, a local late-interaction model with its own on-disk index, for exact names and rare terms. Whichever technique runs, a local cross-encoder then reranks what it found and keeps the 6 Chunks that best answer the question (*reranked* on the answer's Found-with line). Scanned PDF pages with no text are read by the vision model at upload. |
| **Analysis Agent** | Answers questions about your Datasets (CSV, or each sheet of an XLSX). It writes read-only SQL and shows you the query it ran. When SQL isn't enough (statistics, forecasts, charts) it writes Python that runs in a locked-down Docker **Sandbox**, and the charts appear in the chat. |
| **Task Agent** | Reads your Google Calendar and Gmail freely. Anything that changes something or reaches another person (send, reply, create, move or delete an event) waits for you on a **Confirmation card**, where you approve, edit the fields, or reject with a note. |

Around the Agents:

- **Chaining:** one request can use several Agents. A Turn stops after about 6 steps and says so.
- **Threads:** every conversation is saved and resumable from the sidebar.
- **Voice and images:** hold-to-record push-to-talk (Perry reads its answer aloud), and you can drop images or screenshots into the chat.
- **Guardrails:** every model call is recorded with its tokens and cost. A Turn stops at a token ceiling and reports what it found so far; a daily cap refuses new Turns until midnight; bulk uploads show an estimated cost and wait for your OK. A step that fails is retried once, then reported honestly, and a failed Write Action is never retried.
- **Privacy:** a real Google account is refused unless text generation runs on a local model, and tracing is switched off while one is connected. See [Connecting Google](#connecting-google-task-agent).

## Setup

Needs Docker Desktop running and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
git config core.hooksPath .githooks  # ruff + pytest before each commit
docker compose up -d --wait          # Postgres + pgvector on 127.0.0.1:5433
```

Then create `.env` in the repo root (it's gitignored, never commit it):

```ini
GEMINI_API_KEY=...                   # https://aistudio.google.com/apikey (the free tier is enough)
OPENAI_API_KEY=...                   # optional, paid: answers when Gemini's free quota is used up, and runs the benchmark grid
PERRY_SESSION_SECRET=...             # signs login sessions: uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
PERRY_NGROK_DOMAIN=...               # only to serve over ngrok: your free static domain, e.g. your-name.ngrok-free.app
PERRY_INET_QWEN_BASE_URL=...         # optional: the company's model (inet-qwen); set, it does knowledge work first and describes Datasets
# PERRY_DATASET_DESCRIPTION_MODEL=local  # describe Datasets on the local qwen3:8b instead of inet-qwen (the default without a base URL)
# Optional LangSmith tracing; the endpoint must match your account's region:
# api. (US), eu.api. or apac.api.smith.langchain.com (a 403 on every call means a region mismatch)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=...
```

Install [Ollama](https://ollama.com) and pull the local models: embeddings and Summaries run on this machine.

```powershell
ollama pull embeddinggemma-2   # embeddings for every Chunk and Summary (needed)
ollama pull qwen3:8b           # Summaries, Document metadata and RAPTOR (background); Dataset descriptions too without inet-qwen
```

Create yourself as the Owner, once. It asks for a username, a password of at least 10 characters (twice), a display name and a time zone. Each answer is checked as you give it and asked again if it is wrong, so nothing you typed is lost. Enter accepts the suggestion in brackets: your username as the display name, and this machine's time zone (a name like `Asia/Ho_Chi_Minh`; matching ignores case, and a wrong name gets close matches). Ctrl+C stops with nothing created. Threads from before there were Users become yours:

```powershell
uv run python -m perry.users create-owner
```

Everything else has a default; see [Settings](#settings) for the optional overrides.

### Which model answers

Each part of Perry asks a short chain of models, the first one first. The chain is set when Perry starts, from `.env`.

| Part | Chain | Needs |
|---|---|---|
| Knowledge Agent, its techniques' rewriting calls, reading scanned pages | `inet-qwen` → Gemini → OpenAI | `PERRY_INET_QWEN_BASE_URL`; without it, the chain below |
| Supervisor, Analysis Agent, Task Agent | Gemini → OpenAI | |
| Embeddings, Summaries, ColBERT, reranking | local (Ollama and local models), free | |
| Speech | Gemini only | |

- **Gemini** is the free tier, `PERRY_LLM_MODEL` (default `gemini-3.5-flash-lite`).
- **OpenAI** is the paid fallback, `PERRY_FALLBACK_LLM_MODEL` (default `gpt-5.4-mini`). Without `OPENAI_API_KEY` it is left out of every chain.
- **`inet-qwen`** is the company's self-hosted model: free, and Document text stays inside the company. Its server takes no tool calls, so each step asks it for a JSON choice instead (a tool with its arguments, or the answer).

**When a call moves to the next model.** Every model call starts again at the head of its chain. A call moves on when:

- `inet-qwen` can't be reached, returns a server error, takes longer than `PERRY_CHAT_TIMEOUT` (default 45 seconds) or replies with an invalid JSON choice.
- Gemini hits a rate limit (with no retry), stays overloaded after 3 retries, or takes longer than `PERRY_CHAT_TIMEOUT`. Once its daily quota is used up, Gemini is skipped until midnight Pacific time (14:00 or 15:00 in Bangkok). The Turn in which the quota ran out says so, and suggests another Gemini model for `PERRY_LLM_MODEL`, since each model has a free quota of its own.
- OpenAI is the last model, so its errors are reported in the chat. With no OpenAI key, Perry says Gemini's quota is used up, when it comes back and which model to switch to.

A bug in Perry never moves a call on: it is reported. The answer's "Found with" line names the model that answered, e.g. `Gemini (inet-qwen unavailable)`.

**Switching on purpose.** Change `.env`, then restart Perry:

| To | Set |
|---|---|
| Move the Supervisor or an Agent to `inet-qwen` first | `PERRY_INET_QWEN_FOR` lists the parts: `knowledge` (the default), `techniques` (only the techniques' rewriting calls), `analysis`, `task`, `supervisor`, or `all`. Move a part only once the [eval set](#eval-set) shows it does as well there. |
| Move a part back to Gemini → OpenAI | Leave the part out of `PERRY_INET_QWEN_FOR`. |
| Skip Gemini and go straight to OpenAI | `PERRY_PAID_CHAT_ONLY=true`. This applies to every part, and every call is paid. `inet-qwen` still goes first for the parts that use it. |
| Change the Gemini or OpenAI model | `PERRY_LLM_MODEL` or `PERRY_FALLBACK_LLM_MODEL`. |
| Stop using `inet-qwen` | Remove `PERRY_INET_QWEN_BASE_URL`. |

Knowledge work stays on `inet-qwen` even though its eval gate missed by one check (one comparing question was answered from a single Document), because it is the company's own model. The chains are built in `perry/models.py` (`model_for`, `free_first`, `_Chain`). See `docs/adr/0006` and `0007`.

**Dataset descriptions:** what a Data Dictionary leaves out of a Dataset is described in the background by the model `PERRY_DATASET_DESCRIPTION_MODEL` names: `inet-qwen` (the default when `PERRY_INET_QWEN_BASE_URL` is set) or `local` (the local summary model, `qwen3:8b`, the default otherwise). Either way there is no fallback: a Dataset whose describing fails, for example because the model can't be reached, stays usable and is retried on the next start. See `docs/adr/0008`.

## Use

```powershell
uv run python -m perry.web              # Perry at http://127.0.0.1:7860: log in, then chat
uv run python -m perry.ingest some\folder  # bulk-load Documents (.pdf, .md, .txt); shows the estimated cost and asks first (--yes skips the question)
uv run python -m perry.library list   # list, replace, rename or delete Documents (see "Replace, rename or delete a Document")
```

Your data lives in the Postgres from `compose.yaml`, in the Docker volume `pgdata`, so it survives restarts. Perry stores each Document as page-numbered Chunks with their embeddings (not the original file), and each Dataset as a table.

### The UI

Perry asks you to log in first, even when it runs on your own machine; a login lasts 14 days in that browser. *Log out* is at the top of the sidebar. *Profile* there lets you change your display name, time zone and an optional about-me line; Perry reads them on every Turn, and works out "today" or "this week" in your time zone.

To let someone else in, open the **Owner panel** (the link next to your name in the sidebar, or `/owner`) and click *Create an Invite*. Send them the Invite's register link. Each Invite works once and expires after 7 days, and you can withdraw an unused one. On the register page they choose a username, a password (at least 10 characters), a display name and a time zone, and are logged in straight away.

*Log out* logs you out in every browser at once. After 5 failed logins for a username within 15 minutes, logins with it are refused until the 15 minutes are up. A User who forgot their password gets a Reset Code from the Owner panel. If you, the Owner, are locked out or forgot yours, reset it on this machine; it asks for the new password (twice; a short one or a mismatch is asked for again), logs you out everywhere and clears the failed logins:

```powershell
uv run python -m perry.users reset-password <username>
```

- **Sidebar:** *New thread* and a *Threads* picker to resume an earlier conversation; *Delete thread* deletes the open one after you confirm in a pop-up (its usage stays in the Owner's spending figures); *Documents* and *Datasets* upload areas (the Datasets list shows each name with its columns); today's usage (`Today: N tokens · $x of $cap`); and *Calendar and email*, where you pick **Test account** or **Real account** and click **Connect Google**.
- **Chat:** type in the box (the paperclip attaches images). Below it, **Record** is push-to-talk and the player on the right speaks Perry's answer.
- When Perry needs your approval, a **Confirmation card** appears under the chat with the editable fields, **Approve**, **Reject** and a note box.

### Share Perry over ngrok

To let other Users reach Perry from the internet, serve it through an [ngrok](https://ngrok.com) tunnel on your free static domain. Once:

1. Sign up at ngrok, install it (`winget install ngrok.ngrok`) and connect your account: `ngrok config add-authtoken <your token>` (the token is on the dashboard's *Your Authtoken* page).
2. Claim your free static domain on the dashboard's *Domains* page (one per free account) and put it in `.env` as `PERRY_NGROK_DOMAIN`, e.g. `PERRY_NGROK_DOMAIN=your-name.ngrok-free.app`.
3. Make sure `.env` has `PERRY_SESSION_SECRET` and that you've created the Owner (see [Setup](#setup)).

Then, from the repo root:

```powershell
uv run python -m scripts.serve
```

It starts Postgres (`docker compose up -d`, unless it's already up), checks that Ollama answers, launches Perry on 127.0.0.1:7860 (never on a public interface), opens the tunnel with `ngrok http --url=$PERRY_NGROK_DOMAIN` and prints the public URL, `https://<your domain>/`. It refuses to start, saying what's missing, without `PERRY_NGROK_DOMAIN`, `PERRY_SESSION_SECRET`, ngrok, Ollama or an Owner. Ctrl+C stops Perry and ngrok.

Served this way the login cookie is marked Secure, so log in through the public URL; plain `uv run python -m perry.web` is for using Perry on this machine only. ngrok's free plan shows visitors a one-time warning page before Perry; click *Visit Site*. Every Turn anyone takes spends your Gemini quota and Daily Cap.

## Tutorials

The repo ships small sample files you can use to try everything: [evals/documents/](evals/documents/) (four short Documents) and [evals/datasets/sales.csv](evals/datasets/sales.csv) (2025 ad spend, units and revenue for three regions). For questions that join several tables, [sample_data/company/](sample_data/company/) holds a synthetic Vietnamese company: about 100 employees and a year of payroll, as 13 Datasets, each with its Data Dictionary. Answers come from the real model, so expect each one to take a minute or two on the free tier.

### 1. Ask your Documents

1. Start Perry, pick a **Reader** and a **Chunking Strategy** under **Documents** in the sidebar (they start at `PERRY_DOCUMENT_READER` and `PERRY_CHUNKING`), and drop a few files into **Upload** (or run `uv run python -m perry.ingest evals\documents`, which skips the Review).
2. Perry shows an estimated token count and cost and asks whether to read the files for Review. Click **Read for Review**. If an upload would exceed today's remaining budget, it is refused before any call is made.
3. The Review opens above the chat: a table of each Document as read (pages, Chunks, median length, most tokens in a Chunk, pages that couldn't be read, a fingerprint of the Chunks), then a **Page** picker with every page, those that came out with no Chunks too. Each page shows as read beside its Chunks, exactly as they will be embedded. Click **Approve and index** to store them, or **Discard** to store nothing. Nothing is searchable until you approve; the vision Reader's page reading is paid for when you click Read for Review, whichever you choose.
4. Ask, for example: *What's my Q3 marketing budget?* With the sample files you should get 40,000 dollars, with a Citation to `q3-budget.md`.
5. Ask something that isn't in your files, such as *What's my Q4 budget?* Perry should say it can't find it rather than guess.
6. Questions that need two files work too: *How does the offsite compare with the trade fair?*
7. Each answer ends with its Citations, then a line such as `Found with: Hybrid search (picked logically)` naming the Retrieval Technique that found them and how it was picked: `semantically` when the question was clearly closest to that technique's example questions (no extra model call), `logically` when the model chose from the techniques' descriptions. To try a specific technique, pick it in the sidebar's **Retrieval Technique** dropdown (Auto by default); it is used for your next questions, reported as `(forced)`, until you set it back to Auto. Perry still decides on its own whether a question needs your Documents at all.

### 2. Replace, rename or delete a Document

- **Same file name:** upload it again. Perry replaces the old Document and its Chunks in one step. Identical content is skipped at no cost.
- **Anything else** goes through the command line. It works while Perry runs, and Perry's searches see the change at once:

  ```powershell
  uv run python -m perry.library list                               # every Document, its Chunks and upload date
  uv run python -m perry.library replace report.pdf C:\path\report-v2.pdf   # a newer version under another name
eport-v2.pdf   # a newer version under another name
  uv run python -m perry.library rename report.pdf "Q3 report.pdf"  # cited by the new name from now on
  uv run python -m perry.library delete report.pdf                  # asks first; --yes skips the question
  ```

  Names are matched in any case, and accents (e.g. `Báo cáo.pdf`) are matched however they were typed. A misspelt name gets the closest names suggested. Replace shows the estimated cost before it uploads.

  - Deleting or replacing removes the old Document's Chunks, Summary and RAPTOR tree, and its entries in the ColBERT index, in one step. If anything fails, it stays as it was.
  - Afterwards the command rebuilds RAPTOR's cross-Document level on the local model (a minute or more on a CPU). Ctrl+C leaves that to Perry's next start, and until then RAPTOR searches each Document's own tree.
  - Past Threads keep the text they quoted from a deleted Document.
- Scanned PDF pages with no text are transcribed by the vision model and cited by page like any other. The upload estimate includes them, and a page that fails to transcribe is listed after the upload.

### 3. Analyse a Dataset

1. Drop `sales.csv` (or your own CSV/XLSX; each sheet becomes its own Dataset) into **Datasets**. It appears in the list with its columns. Uploading a file with the same name replaces that Dataset.
2. Ask: *What was total 2025 revenue per region?* Perry runs a read-only SQL query and shows the query it used, so you can check the logic. With the sample file the totals are North 181,300, South 156,350 and West 220,850.
3. Ask for something SQL is bad at: *Is ad spend correlated with revenue?* or *Forecast the next three months of revenue.* Perry writes Python, runs it in the Sandbox and shows the code it used.
4. Ask for a picture: *Plot monthly revenue per region as a chart.* The PNG appears in the chat above the answer.
5. A bad or slow query is cut off by a timeout and a row limit, and Perry tells you what failed so you can rephrase.

The first Python question builds the Sandbox image, which takes about 2 minutes and needs Docker running.

### 4. Calendar and email (test account first)

1. Follow [Connecting Google](#connecting-google-task-agent) once. In the sidebar choose **Test account (fake data only)**, then **Connect Google**.
2. Read-only questions run straight away: *What's on tomorrow?*, *What did Sam say about the budget last week?*, *Summarise my latest email thread with Sam.*
3. Ask for a change: *Email Sam a short summary of the Q3 budget.* Perry pauses on a **Confirmation card** showing exactly what will happen (To, Subject, Message).
4. Choose one:
   - **Approve** performs it exactly once.
   - **Edit** any field first (recipients, subject, body, start and end times, invitees), then **Approve**. What you edited is what gets performed.
   - **Reject** with a note in the box (for example "make it shorter and friendlier") and Perry revises and asks again.
5. If you type a new message while a card is waiting, the waiting action is cancelled and Perry says so. Nothing is sent hours later by surprise.
6. If a Write Action fails, Perry reports it and never retries, so you never get duplicate emails or events. If it says the outcome is unknown, check your account before asking again.

### 5. Chain several Agents

Ask for one result that needs more than one Agent: *Find my Q3 budget notes and email Sam a summary.* The Knowledge Agent finds the notes with Citations, then the Task Agent drafts the email and pauses on a Confirmation card before anything is sent. A chain never skips your approval.

### 6. Talk to Perry

Click **Record**, speak, and click stop. Your words appear as your message so you can catch mishearings, then Perry answers in text and reads the answer aloud. Speech counts toward your usage limits. If the daily cap is already reached, Perry says so instead of transcribing.

### 7. Ask about an image

Click the paperclip (or drop a file) to attach a screenshot, chart or receipt, and ask: *What does this error mean?* The image stays in the Thread, so follow-up questions about it work.

### 8. Keep an eye on cost

- The sidebar line `Today: N tokens · $x of $0.50` is read from the usage ledger.
- Every model call is in the ledger with what it was for, e.g. `rewrite`, `decompose`, `hyde`, `summary`, `raptor` or `colbert`. Semantic picking's embeddings are recorded as `pick`; logical picking has no call of its own (the Knowledge Agent's model picks while answering, recorded as `chat`).
- A single Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far.
- Once `PERRY_DAILY_CAP_USD` (default $0.50) is reached, new Turns are refused until local midnight. A Turn already running always finishes.
- Prices live in `prices.toml`; edit it to match current pricing. All limits are set in `.env`.

## More about the features

**Datasets** are stored as real tables in an `analytics` schema. The Analysis Agent's SQL runs as a separate Postgres role (`perry_analyst`) that can only read that schema, one statement at a time, in a read-only transaction with a statement timeout and a row limit, so a bad query can't touch your Documents, Threads or credentials.

The **Sandbox** is a throwaway Docker container from the image in `sandbox/Dockerfile` (pandas, numpy, scipy, matplotlib), built on first use. It has no network, CPU, memory and time limits, no environment from the host, and sees only a temp folder holding the input CSVs Perry wrote from read-only queries. Charts it saves are kept under `%APPDATA%\Perry\charts`. Sandbox tests skip when Docker isn't running.

Push to talk uses the chat model for transcription and `PERRY_TTS_MODEL` for speech.

## Eval set

```powershell
uv run python -m evals --list            # the eval Turns
uv run python -m evals                   # run them all against the real model
uv run python -m evals knowledge-aphids  # only Turns whose id starts with this
uv run python -m evals --tune-margin     # tune the cascade margin (embeddings only)
uv run python -m evals --compare         # the latest two saved runs against the switch-over gate
```

Each eval Turn ([evals/turns/](evals/turns/), one file per Agent) says which Agents must be consulted, which Citations the answer must contain and which Write Actions must wait for a Confirmation (the run approves them, against the fake); those checks pass or fail the run. An LLM judge also scores answer quality, which is only reported (and logged to the LangSmith project `perry-eval` when tracing is on). The run uses its own `perry_eval` database loaded from [evals/documents/](evals/documents/) and [evals/datasets/](evals/datasets/) (the `analysis-python-*` Turns run Python in the Sandbox, so they need Docker), and a fake calendar and mailbox ([evals/workspace.py](evals/workspace.py)) instead of Google. It isn't part of `uv run pytest` or the pre-commit hook, because it spends real requests and takes several minutes.

Knowledge Turns can also name the Retrieval Technique expected to find the answer ([evals/turns/techniques.py](evals/turns/techniques.py) has at least two per technique). That check is soft: a mismatch is printed and logged to LangSmith as `technique` feedback, but never fails the run. The run forces the "Found with" line on, since that is where it reads the technique from.

**Saving and comparing runs.** Every run saves its results to [evals/runs/](evals/runs/) (kept out of git): when it ran, the commit, who answered (chat and fallback models, `inet-qwen` and `PERRY_INET_QWEN_FOR`), and per eval Turn its routing, Citation and Confirmation checks, judge score and cost (the Turn's model calls and its judge's, from the Usage Ledger). `--compare BASELINE CANDIDATE` (or no files: the latest two) compares two runs on the Turns both ran, saying how many were left out. It prints each run's routing and Citation checks passed, average judge score and the difference, its cost, the Turns whose result differs (a check that flipped, Confirmations included, or a judge score a point or more apart), and the verdict against the switch-over gate of [ADR-0006](docs/adr/0006-knowledge-work-on-the-company-model.md): at most one fewer routing and Citation check passed, and an average judge score at most 10% lower. It exits 1 when the candidate fails. To gate a part moving to `inet-qwen`, run the eval set without it and then with it added to `PERRY_INET_QWEN_FOR`, both with `--paid` and `PERRY_BENCHMARK_LLM_MODEL=gpt-5.4-mini` so the same model judges both, then `--compare`.

**Tuning the cascade margin.** `--tune-margin` embeds every eval Turn that expects a technique (embedding calls only, a fraction of a cent) and measures how much nearer semantic picking's top technique is than the runner-up, and whether the top one is the expected one. The tuned margin is the smallest that sits above every wrong pick's gap (midway to the next right one): every question cascade settles semantically on the eval Turns was picked right, and the rest go to logical picking, which is assumed right there (measuring it needs chat calls; the full eval run's soft check does that). Record the printed value as `DEFAULT_CASCADE_MARGIN` in [perry/picking.py](perry/picking.py) and in the settings table. Rerun it after changing technique example questions, technique eval Turns or the embedding model. Eval questions must never be benchmark questions or router example questions (a test checks), or the tuning grades its own answer key.

The current `0.1425` comes from a run on 2026-11-13 with `embeddinggemma-2`: semantic picking alone was right on 10 of 22 Turns, and its most confident wrong pick sets the margin above every Turn's gap, so for now cascade leaves every question to logical picking (never a wrong semantic pick, at one extra model call each). Better example questions would let cascade settle more questions semantically. (With `nomic-embed-text` it was `0.116`, 8 of 22 right; with `gemini-embedding-001`, `0.074`, 11 of 22 right.) Rerun the tuning after changing the embedding model or the example questions; it only embeds, so it's free.

## Connecting Google (Task Agent)

The Task Agent reads your calendar and email freely, and performs Write Actions (send, reply, create, move or delete an event) only after you approve their Confirmation card. A new message cancels a waiting Confirmation; a failed Write Action is reported, never retried. Use a **test Google account containing only fake data**: on the free tier, what Perry reads is sent to Gemini.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project, enable the **Google Calendar API** and **Gmail API**, and set up the OAuth consent screen (External). Add the scopes `calendar.events`, `gmail.readonly` and `gmail.send`, then **publish the app to production** so sign-ins don't expire after 7 days (an unverified personal app is fine; you click through the warning).
2. Create an OAuth client of type **Desktop app**, download its JSON and save it as `%APPDATA%\Perry\google_client.json` (or point `PERRY_GOOGLE_CLIENT_FILE` at it). Never put it in the repo.
3. Start Perry, choose **Test account** or **Real account** in the sidebar, and click **Connect Google**. A browser tab asks for consent; afterwards the button reads "Google connected".

Privacy switches, enforced in code at connect time and at startup:
- A **real account** is refused unless text generation runs on a local model (for now it's Gemini, so only test accounts connect). If a real account is already connected while a cloud model is configured, the Task Agent is off and says why.
- While a real account is connected, LangSmith tracing is switched off, whatever `.env` says.
- A sign-in kept before Perry asked for the kind counts as real: reconnect and choose **Test account**.

The sign-in is kept in `%APPDATA%\Perry\google_token.json`, outside the repo, Postgres and the sandbox. If Google refuses it (expired or revoked), Perry answers "Google needs reconnecting" and the button turns into **Reconnect Google**.

Manual check against the test account: add a few events for tomorrow and a few emails (e.g. from "Sam" about a budget), then ask "What's on tomorrow?" and "What did Sam say about the budget?". To check the reconnect path, revoke Perry at <https://myaccount.google.com/permissions> and ask again.

## Knowledge benchmark (ADR-0001)

`benchmark/` runs the same 23 questions against Perry and the predecessor QA-bot (LightRAG) over the Documents in `benchmark/documents/`, and scores correctness and Citation accuracy per question type. See the docstring of `benchmark/run.py` for the commands; results land in `benchmark/results/`.

### Technique grid

The grid asks the same 23 questions with every Retrieval Technique forced (10 setups) and every picking mode (`semantic`, `logical`, `cascade`), so you can see which technique wins on which question type. Always-hybrid (Perry before Retrieval Techniques) is exactly `hybrid_search` forced, so it is that row, run and paid for once. It runs in its own database, `perry_benchmark`, so it never touches your Documents or your Daily Cap.

```bash
docker compose up -d
ollama serve            # embeddings, plus Summaries and RAPTOR trees (free but slow)
uv run python -m benchmark.run grid
```

The full grid is 299 Knowledge Agent Turns, every one on the paid OpenAI model `PERRY_BENCHMARK_LLM_MODEL` (default `gpt-5.4-nano`), so its rows never mix two models; it needs `OPENAI_API_KEY`. With `PERRY_INET_QWEN_BASE_URL` set, knowledge work runs on `inet-qwen` instead and the rest on that model; each row's `model` lists both. It prints the estimated cost of the Turns still to ask (about $1 for the full grid on nano) and waits for your OK; `--yes` skips the question. Before asking anything, the run gets the benchmark Documents ready: it ingests them, brings the ColBERT index in step with Postgres, and, when a setup may search Summaries (`document_summaries`, `raptor` or any picking mode), writes every Summary and RAPTOR tree on the local model. That takes minutes per Document on a CPU, once; later runs reuse them. If Ollama isn't running or a Document fails to summarise, the run stops before spending anything and says why.

Limit a run to some setups, e.g. after changing one technique:

```bash
uv run python -m benchmark.run grid --only hyde,cascade           # fill in missing rows
uv run python -m benchmark.run grid --only hyde --rerun           # ask hyde afresh
uv run python -m benchmark.run grid-report                        # redraw the table only
```

`--only` takes technique names, picking modes and `always-hybrid`. Other setups' rows are kept. Plain setups run with Reranking off; add `+rerank` to run a setup with it (ADR-0005), e.g. to compare the two:

```bash
uv run python -m benchmark.run perry-retrieval   # free: page recall of hybrid search, without and with Reranking
uv run python -m benchmark.run perry-retrieval --reader docling,pymupdf,vision --chunking heading,fixed
    # the same for each Reader and Chunking Strategy -> results/perry-retrieval-comparison.md
    # (vision reads every page on gpt-5.4-mini: it shows the estimate and asks first)
uv run python -m benchmark.run grid --only hybrid_search,hybrid_search+rerank,cascade,cascade+rerank   # 92 Turns, about $0.26
```

Rows are saved to `benchmark/results/grid.jsonl` as they arrive, so a run that stops (e.g. quota) resumes where it stopped; `--rerun` drops the selected setups' rows first.

**Reading the table** (`benchmark/results/GRID.md`): *Overall* has one line per setup, with how many questions it answered, *Correct* (every expected fact present), fact recall, *Citation recall* (share of required pages cited), *Citation precision* (share of cited pages that are right), cost per question and in total (from the Usage Ledger: paid calls only cost; local ones are free), and the model and run date of its rows. A setup whose date is older than a change you made is stale: rerun it. The *per question type* tables show the same measures for lookup, multi-hop, thematic and entity questions, which is where techniques differ. *Per question* shows each setup's fact recall per question, with `c` when every required page is cited. Picking-mode rows also record, in `grid.jsonl`, which technique the mode chose for each question (`found_with`).

## Testing a change

See [docs/testing.md](docs/testing.md): the test suite, live questions with `uv run python -m scripts.ask` (each Turn's time, and every model call's tokens and cost, recorded in `scripts/results/asks.jsonl`), forcing each model fallback, and the eval set.

## Settings

All optional; add any of these to `.env` to override the default.

| Setting | Default | What it does |
|---|---|---|
| `PERRY_NGROK_DOMAIN` | none | Your ngrok static domain; `scripts.serve` needs it (see [Share Perry over ngrok](#share-perry-over-ngrok)) |
| `PERRY_LLM_MODEL` | `gemini-3.5-flash-lite` | Chat model (free tier); also used for speech-to-text and image reading |
| `PERRY_FALLBACK_LLM_MODEL` | `gpt-5.4-mini` | Paid OpenAI chat model used once Gemini's free quota is used up or it stays overloaded (needs `OPENAI_API_KEY`); give it a price in `prices.toml` |
| `PERRY_INET_QWEN_BASE_URL` | *(unset)* | The company's model server (e.g. `https://ai-api.inetcloud.vn/knowledge/v1`); set, knowledge work runs on it first |
| `PERRY_INET_QWEN_FOR` | `knowledge` | What asks `inet-qwen` first, comma-separated: `knowledge` (the Knowledge Agent and its techniques' helper calls), `techniques` (only the helper calls), `analysis`, `task`, `supervisor`, or `all` |
| `PERRY_INET_QWEN_MODEL` | `knowledge` | The model name on that server; give it a price in `prices.toml` (0) |
| `PERRY_INET_QWEN_API_KEY` | *(unset)* | Only if the server ever asks for a key; a placeholder is sent otherwise |
| `PERRY_DATASET_DESCRIPTION_MODEL` | `inet-qwen` with `PERRY_INET_QWEN_BASE_URL` set, else `local` | What describes, in the background, the Datasets and columns a Data Dictionary leaves out (marked as the model's guesses): `inet-qwen`, the company's model, or `local`, the local summary model (`PERRY_SUMMARY_MODEL`). No fallback: a Dataset it fails on is retried on the next start. Any other value, or `inet-qwen` without a base URL, stops Perry at start |
| `PERRY_PAID_CHAT_ONLY` | `false` | `true`: chat goes straight to the paid model, never Gemini (faster, but every Turn costs money; needs `OPENAI_API_KEY`). Knowledge work still runs on `inet-qwen` first when it is set |
| `PERRY_BENCHMARK_LLM_MODEL` | `gpt-5.4-nano` | Paid OpenAI chat model every benchmark-grid Turn runs on |
| `PERRY_OPENAI_REASONING_EFFORT` | `none` | Reasoning effort for the OpenAI models (`none` is the cheapest, and the only one the API accepts with tools) |
| `PERRY_EMBEDDING_MODEL` | `embeddinggemma-2` | Local Ollama embedding model; changing it re-embeds every Chunk and Summary when Perry next starts; give it a zero entry in `prices.toml` |
| `PERRY_EMBEDDING_TOKENIZER` | `google/embeddinggemma-2` (`nomic-ai/nomic-embed-text-v1.5` for nomic) | The embedding model's tokenizer on Hugging Face (downloaded on first use), which measures heading Chunks; empty estimates 4 characters a token |
| `PERRY_CHUNKING` | `heading` | Chunking Strategy for uploads (docs/adr/0009): `heading` keeps each Chunk in one section, led by its headings (at most 384 tokens); `fixed` cuts each page into overlapping pieces of about 1,500 characters. A file uploaded again under the other strategy is chunked again |
| `PERRY_DOCUMENT_READER` | `docling` | Reader for PDFs (docs/adr/0009): `docling` (layout, headings, tables; local), `pymupdf` (the text layer only; local, fastest), or `vision` (`PERRY_PAGE_READER_MODEL` reads every page as Markdown; paid, needs `OPENAI_API_KEY`). A file uploaded again under another Reader is read again |
| `PERRY_PAGE_READER_MODEL` | `gpt-5.4-mini` | OpenAI vision model the `vision` Reader reads pages with; give it a price in `prices.toml` |
| `PERRY_EMBEDDING_DIMENSIONS` | `768` | The embedding model's vector size; changing it resizes the columns and re-embeds at the next start |
| `PERRY_TTS_MODEL` | `gemini-2.5-flash-preview-tts` | Reads voice answers aloud |
| `PERRY_TTS_VOICE` | `Kore` | Voice for read-aloud answers |
| `PERRY_SUMMARY_MODEL` | `qwen3:8b` | Local Ollama model that writes Summaries and extracts Document metadata, in the background, and describes Datasets when `PERRY_DATASET_DESCRIPTION_MODEL` is `local`; give it a zero entry in `prices.toml` |
| `PERRY_OLLAMA_URL` | `http://127.0.0.1:11434` | Where Ollama listens |
| `PERRY_SUMMARY_CONTEXT_TOKENS` | `8192` | Context window asked of the local model (Ollama's default, 4096, cuts longer prompts) |
| `PERRY_CHAT_TIMEOUT` | `45` | Seconds a free chat call may take before the paid model answers instead, and an `inet-qwen` call before Gemini does |
| `DATABASE_URL` | `postgresql://perry:perry@127.0.0.1:5433/perry` | The Postgres Perry uses |
| `PERRY_RETRY_DELAY_SECONDS` | `2` | Wait before a failed step's one retry |
| `PERRY_TURN_TOKEN_CEILING` | `50000` | A Turn stops and reports its findings past this many tokens |
| `PERRY_DAILY_CAP_USD` | `0.50` | New Turns and ingestion are refused past this until local midnight |
| `PERRY_COLBERT_MODEL` | `answerdotai/answerai-colbert-small-v1` | Local model for the ColBERT Retrieval Technique (downloaded on first use, runs on the CPU); changing it re-encodes every Chunk at next start; empty turns ColBERT off |
| `PERRY_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local cross-encoder for Reranking every search's Chunks (downloaded on first use, runs on the CPU, English); `BAAI/bge-reranker-v2-m3` for other languages (slower); empty turns Reranking off |
| `PERRY_RERANK_MIN_SCORE` | (off) | If set, Chunks the reranker scores below it are left out (the best one is always kept). Off by default: the default model scores some right answers very low (see ADR-0005) |
| `PERRY_SHOW_FOUND_WITH` | `true` | End answers from your Documents with the "Found with" line naming the Retrieval Technique; `false` turns it off |
| `PERRY_PICKING_MODE` | `cascade` | How the Knowledge Agent picks a Retrieval Technique: `semantic` (nearest example questions), `logical` (the model chooses from the descriptions) or `cascade` (semantic when clearly closest, else logical). The sidebar's forced technique wins over all of them |
| `PERRY_CASCADE_MARGIN` | `0.1425` | How much nearer (cosine similarity) the closest technique's examples must be than the runner-up's for `cascade` to pick semantically; tuned on the eval set, never the benchmark (see [Eval set](#eval-set)) |
| `PERRY_PRICES_FILE` | `prices.toml` | Price table, USD per million tokens per model |
| `PERRY_ANALYST_PASSWORD` | `perry_analyst` | Password of the read-only `perry_analyst` role |
| `PERRY_ANALYSIS_TIMEOUT_MS` | `10000` | Statement timeout for the Analysis Agent's SQL |
| `PERRY_ANALYSIS_ROW_LIMIT` | `200` | Rows returned per SQL query |
| `PERRY_SANDBOX_TIME_LIMIT_S` | `60` | Sandbox time limit per script |
| `PERRY_SANDBOX_MEMORY` | `1g` | Sandbox memory limit |
| `PERRY_SANDBOX_CPUS` | `1` | Sandbox CPU limit |
| `PERRY_SANDBOX_INPUT_ROW_LIMIT` | `100000` | Rows per input file written for a script |
| `PERRY_APP_DATA_DIR` | `%APPDATA%\Perry` | Holds `google_token.json`, saved charts and the ColBERT index (`colbert\`) |
| `PERRY_GOOGLE_CLIENT_FILE` | `<app-data>\google_client.json` | Google OAuth client file |

## Notes

- **Speed on the free tier:** chat isn't paced, since Flash-Lite's free tier allows many more requests a minute than a Turn makes (3-6 model calls). A call that still hits the per-minute limit goes to the paid model, and so costs a little.
- **Privacy:** on the free tier, Google may use what you send to improve its products. Don't upload anything confidential.
- **Scanned PDF pages** (no extractable text) are rendered to images and transcribed by the vision model (the chat model, `PERRY_LLM_MODEL`), then cited by page like any other. The estimate shown before ingesting includes one transcription per such page, and the calls count toward usage. A page whose transcription fails is listed after upload.
- **Summaries (local model):** after an upload returns, a background job asks the local model (Ollama, `PERRY_SUMMARY_MODEL`, default `qwen3:8b`: `ollama pull qwen3:8b`) for one Summary per Document plus its date and topic, reading the start of the Document (about 2,500 tokens). On a CPU this takes minutes per Document; the sidebar shows "Summarising N Documents…" meanwhile, and the Document Summaries technique skips Documents that aren't ready. If Ollama isn't running, the Document stays searchable, is marked failed, and is retried on the next start. The upload estimate counts the local calls as free. The `python -m perry.ingest` CLI doesn't summarise; the app does it for pending Documents when it next starts.
- **ColBERT** keeps its own index under `<app-data>\colbert`, beside Postgres (docs/adr/0002). Every upload updates both before it returns, and a failed upload leaves both as they were. Uploads take a little longer while ColBERT encodes the text on the CPU (roughly 7 Chunks a second), and the first one downloads the model (~130 MB). At start, Perry encodes any Chunks stored before ColBERT existed and rebuilds the index if it doesn't match Postgres; deleting the folder is safe.
- **Changing the embedding model** (`PERRY_EMBEDDING_MODEL`) or its size: at the next start Perry resizes the vector columns if needed and re-embeds every Chunk and Summary with the new model (docs/adr/0003; about 1 s per Chunk on a CPU). Meanwhile, searches still find Chunks by their words.
- **Usage:** every model call is recorded with its tokens and cost (the sidebar shows today's totals). Prices live in `prices.toml`; edit it to match current pricing. A Turn stops at `PERRY_TURN_TOKEN_CEILING` tokens (default 50,000) and reports what it found so far. Embedding tokens are estimated (about 4 characters per token). Prices are what you really pay: free-tier Gemini and local models at 0, the OpenAI models at their list prices, so the Daily Cap counts real money. Once today's cost reaches `PERRY_DAILY_CAP_USD` (default $0.50), new Turns are refused until local midnight; a Turn already running always finishes. Uploads count toward the cap too: they show an estimate and wait for your OK, and are refused up front if they would go over.
