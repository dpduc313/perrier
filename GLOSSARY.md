# Perry

A personal assistant that answers from your own documents and data, and acts on your calendar and email, through one conversation.

## Language

### People

**User**:
A person who logs into Perry with a username and password. Every User searches the same Documents and Datasets, but sees only their own Threads.
_Avoid_: Account (that is the Google account behind the Workspace), member, client

**Profile**:
The basic facts a User gives about themselves, such as name and time zone, which Perry reads when answering them.
_Avoid_: Memory (that would be what Perry learns from conversation), settings

**Owner**:
The one User Perry belongs to. Only the Owner uploads Documents and Datasets, connects the Workspace, creates Invites and Reset Codes, and has Turns that may involve the Task Agent.
_Avoid_: Admin, superuser

**Invite**:
A single-use code from the Owner that lets one person register as a User. It expires after a week.
_Avoid_: Invitation link, sign-up code, token

**Reset Code**:
A single-use code from the Owner that lets a User who forgot their password choose a new one. It expires after a day, and using it logs that User out everywhere else.
_Avoid_: Reset token, reset link, recovery code

### Orchestration

**Supervisor**:
The coordinator that reads a request, decides which Agents to involve and in what order, and composes the final answer.
_Avoid_: Router, orchestrator, master agent

**Agent**:
A specialist with one area of responsibility that the Supervisor delegates to. Perry has exactly three: the Knowledge Agent, the Analysis Agent and the Task Agent.
_Avoid_: Worker, sub-bot, skill

**Knowledge Agent**:
The Agent that finds relevant Chunks in your documents and answers with Citations.
_Avoid_: Retrieval agent, RAG agent, search agent

**Analysis Agent**:
The Agent that answers questions by querying and computing over Datasets.
_Avoid_: Data agent, SQL agent

**Task Agent**:
The Agent that reads from and acts on your calendar and email.
_Avoid_: Action agent, execution agent

**Turn**:
One request from you and everything Perry does to answer it, possibly involving several Agents in sequence.

**Task Decomposition**:
The Supervisor splitting one request into steps for different Agents, such as finding notes with the Knowledge Agent and then emailing them with the Task Agent.
_Avoid_: Decomposition (on its own), planning, query decomposition

**Thread**:
A resumable conversation made of Turns, belonging to one User.
_Avoid_: Session, chat, conversation history

**Usage Ledger**:
The record of every model call (chat, embeddings, and later speech and vision) with its Thread, Turn, kind, token counts and cost, priced from the price table in config.
_Avoid_: Billing log, token log

**Token Ceiling**:
The most tokens one Turn may use. A Turn that reaches it stops and reports what it found so far.
_Avoid_: Token limit (ambiguous with the model's context window), budget

**Daily Cap**:
The most money Perry may spend per day, counted from the Usage Ledger since local midnight. Once reached, new Turns and ingestion are refused until midnight; a Turn already running finishes.
_Avoid_: Quota (that is the provider's limit), daily limit

### Knowledge

**Document**:
A file you have uploaded so Perry can search it. A newer upload with the same name replaces the older one; a newer version under another name can replace it too. Deleting a Document removes everything Perry learned from it, but past Threads keep what they quoted.
_Avoid_: File, source, attachment

**Chunk**:
A passage of a document that can be retrieved on its own and traced back to its page.
_Avoid_: Snippet, passage, fragment

**Chunking Strategy**:
How a Document is split into Chunks: by heading (each Chunk stays inside one section and one page, led by its headings) or by fixed length (each page cut into overlapping pieces of the same size). A setting chooses it for every upload, and the Owner can pick the other for an upload in the chat UI; a Document records the one that split it.
_Avoid_: Chunker, splitter, chunking mode

**Reader**:
What turns a PDF into text before it is chunked: Docling (layout, headings and tables), PyMuPDF (the text layer alone), or a vision model reading each page as an image. A setting chooses it for every upload, and the Owner can pick another for an upload in the chat UI; a Document records the one that read it.
_Avoid_: Parser, extractor, OCR (that is only part of what a Reader may do)

**Review**:
The Owner's look at an upload after it is read and chunked, before anything is stored: every page as read beside its Chunks, exactly as they will be embedded. Only what the Owner approves is indexed; discarding stores nothing. Uploads in the chat UI always go through one; the ingest CLI skips it.
_Avoid_: Preview, approval step, validation

**Citation**:
A pointer from an answer to the exact document page a claim came from.
_Avoid_: Source, reference

**Retrieval Technique**:
One way the Knowledge Agent can find Chunks for a question, such as rewriting it into several queries, searching with a hypothetical answer, or searching Summaries. Every Retrieval Technique ends in Chunks with Citations.
_Avoid_: RAG technique, retriever, search mode

**Reranking**:
The step after every Retrieval Technique that reads the question with each Chunk the technique found and keeps the few that answer it best, best first. It is not a Retrieval Technique: it finds no Chunks of its own.
_Avoid_: Rerank technique, re-ranker stage, re-scoring

**Question Decomposition**:
The Retrieval Technique that splits one question into sub-questions, finds Chunks for each, and answers from all of them together.
_Avoid_: Decomposition (on its own), task decomposition, sub-querying

**Summary**:
A model-written condensation of a Document, or of a group of related Chunks, used only to find Chunks. A Summary is never cited; answers cite the pages of the Chunks it led to.
_Avoid_: Abstract, digest, overview

**Dataset**:
A table of your structured data (uploaded from CSV or a spreadsheet) that the Analysis Agent can query.
_Avoid_: Table (when talking to users), spreadsheet

**Value Profile**:
What one Dataset column's values look like, worked out from the rows at upload: the share of empty values, the smallest and largest (for numbers and dates), and every distinct value of a short list or a few examples of a long one. The Analysis Agent reads it to filter on values that actually occur.
_Avoid_: Profile (on its own: that is a User's), statistics, column summary

**Data Dictionary**:
A file you upload to say what your Datasets mean: what each Dataset and column holds, in what unit, and which column points at another Dataset. It describes Datasets; it isn't one.
_Avoid_: Schema, metadata file, codebook

**Join Hint**:
That a Dataset column holds values of another Dataset's column, so the Analysis Agent knows how to join them. Declared in a Data Dictionary, or inferred when a column named `x_id` has every value in another Dataset whose first column is `x_id`. A hint guides the SQL; it never refuses data.
_Avoid_: Foreign key, relationship

**Sandbox**:
The throwaway, locked-down container where the Analysis Agent's Python runs: no network, limited CPU, memory and time, and no view of your files, secrets or database. It sees only the data files Perry wrote for that run from read-only queries, and hands back printed text, tables and charts.
_Avoid_: Code interpreter, executor, container (when talking to users)

### Actions

**Workspace**:
The Owner's calendar and email as Perry sees them, through one connected Google account. The Task Agent reaches them only through the Workspace, never Google directly. Email threads in the Workspace are "email threads", never Threads.
_Avoid_: Google, account, integration (when meaning the calendar and email)

**Test account** / **Real account**:
What you declare the connected Google account to be when connecting it: a test account holds only fake data; a real one is yours. A real account is refused unless text generation is local, and is never traced.

**Read Action**:
An action on calendar or email that only looks, such as listing events or fetching messages. Runs without asking.

**Write Action**:
An action on calendar or email that changes something or reaches another person, such as sending, scheduling or deleting.
_Avoid_: Side effect, mutation

**Confirmation**:
Your explicit approval of one specific Write Action, shown exactly as it will be performed and editable before you approve. No Write Action happens without one, and each one covers a single attempt.
_Avoid_: Approval prompt, consent
