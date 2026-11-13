"""Perry's chat UI, served behind the login pages by perry.web. Run with:
uv run python -m perry.web

Handlers that act for the User take Gradio's request first and get the User with
`user_of`, never from anything the browser holds."""

import html
import io
from pathlib import Path

import gradio as gr
import PIL.Image

from perry import (
    charts,
    dataset_descriptions,
    datasets,
    documents,
    google_auth,
    ingest,
    models,
    pages,
    privacy,
    summaries,
    techniques,
    threads,
    usage,
    users,
    workspace,
)
from perry.supervisor import Confirmation

EMPTY = {"text": "", "files": []}

# Logging out is a POST (so a link elsewhere can't log you out): submit a form for it.
LOG_OUT_JS = f"""() => {{
  const form = document.createElement("form");
  form.method = "post";
  form.action = "{pages.LOGOUT_PATH}";
  document.body.appendChild(form);
  form.submit();
}}"""

# The Thread delete pop-up: a dialog over a dimmed page, which blocks the page until the
# User picks Delete or Cancel. Cancel is focused, so Enter cancels.
CSS = """
#delete-thread-popup {
  position: fixed; inset: 0; z-index: 1000;
  display: flex; align-items: center; justify-content: center;
  padding: 16px; background: rgba(0, 0, 0, 0.5);
}
#delete-thread-dialog {
  width: 100%; max-width: 28rem;
  padding: var(--spacing-lg); background: var(--background-fill-primary);
}
/* The Review: a page's Chunks beside its image, each exactly as it will be embedded. */
#review-chunks { max-height: 70vh; overflow-y: auto; }
.review-chunk {
  margin-bottom: var(--spacing-lg); padding: var(--spacing-md);
  border: 1px solid var(--border-color-primary); border-radius: var(--radius-md);
}
.review-chunk-meta { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.review-chunk pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
"""
FOCUS_CANCEL = """() => {
  setTimeout(() => document.querySelector("#delete-thread-cancel")?.focus(), 50);
}"""

LOG_IN_AGAIN = "You're not logged in. Reload the page to log in again."


def user_of(request: gr.Request | None) -> users.User:
    """The User Gradio's request belongs to (perry.web gives Gradio their id as the
    request's username). Refused if there is none or they are disabled."""
    username = getattr(request, "username", None)
    user = users.get(int(username)) if username and str(username).isdigit() else None
    if user is None or user.disabled:
        raise gr.Error(LOG_IN_AGAIN)
    return user


def thread_of(request: gr.Request | None, thread_id: str | None) -> threads.Thread:
    """The request's User's Thread `thread_id` (None: one not started yet). Another
    User's Thread is refused as if it didn't exist."""
    try:
        return threads.open(user_of(request), thread_id)
    except threads.ThreadNotFound as missing:
        raise gr.Error(str(missing)) from None


OWNER_ONLY = "Only the Owner can do that."


def owner_of(request: gr.Request | None) -> users.User:
    """The Owner, if Gradio's request is theirs; refused for anyone else (ADR-0004)."""
    user = user_of(request)
    if not user.is_owner:
        raise gr.Error(OWNER_ONLY)
    return user


# The sidebar's Owner-only controls: Document upload, Dataset upload, Workspace connect.
OWNER_CONTROLS = ("documents", "datasets", "workspace")


def owner_controls(request: gr.Request) -> list:
    """Show the Owner-only controls (hidden until this says otherwise) to the Owner only."""
    shown = user_of(request).is_owner
    return [gr.update(visible=shown) for _ in OWNER_CONTROLS]


def logged_in_as(request: gr.Request) -> str:
    """The sidebar's note on who is logged in, with a link to the Owner panel for the
    Owner."""
    user = user_of(request)
    note = f"Logged in as **{user.profile.display_name}**"
    return f"{note} · [Owner panel](/owner)" if user.is_owner else note


# Every field a Write Action can have: (label, lines, editable). The Confirmation card
# shows the ones its Write Action takes.
CARD_FIELDS = {
    "to": ("To (comma-separated)", 1, True),
    "subject": ("Subject", 1, True),
    "body": ("Message", 6, True),
    "title": ("Title", 1, True),
    "start": ("Start (YYYY-MM-DDTHH:MM)", 1, True),
    "end": ("End (YYYY-MM-DDTHH:MM)", 1, True),
    "location": ("Location", 1, True),
    "attendees": ("Invite (comma-separated emails)", 1, True),
    "email_thread_id": ("Email thread", 1, False),
    "event_id": ("Event", 1, False),
}


def show_user_message(message: dict, history: list[dict]):
    """Show the message (text and any dropped images) in the chat and clear the box."""
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return EMPTY, history, EMPTY
    shown = [{"role": "user", "content": {"path": image}} for image in images]
    if text.strip():
        shown.append({"role": "user", "content": text})
    return {"text": text, "files": images}, [*history, *shown], EMPTY


AUTO = "auto"  # the dropdown's default: the Knowledge Agent picks the Retrieval Technique


def technique_choices() -> list[tuple[str, str]]:
    """The Retrieval Technique dropdown: Auto, then every registered technique."""
    return [("Auto", AUTO), *((t.label, t.name) for t in techniques.TECHNIQUES)]


def answer(
    request: gr.Request,
    message: dict,
    history: list[dict],
    thread_id: str | None,
    technique: str = AUTO,
    heard: threads.Heard | None = None,
):
    """Run the Turn (the one `heard` began, when the request came by voice), forcing
    `technique` unless it is Auto. Outputs: chat, Thread, Thread picker, then `show()`'s
    card outputs."""
    thread = thread_of(request, thread_id)
    text, images = message.get("text", ""), message.get("files", [])
    if not text.strip() and not images:
        return history, thread_id, gr.skip(), *[gr.skip() for _ in confirmation_card(None)]
    # A pending Confirmation is cancelled by this message; the reply says so.
    forced = None if technique in (None, AUTO) else technique
    outcome = thread.send(heard or text, images=images, technique=forced)
    history, *card = show(outcome, history)
    picker = gr.update(choices=threads.list_recent(thread.user), value=thread.id)
    return history, thread.id, picker, *card


def show(outcome: str | Confirmation, history: list[dict]):
    """What a Turn's outcome looks like: the chat, then the Confirmation card's outputs
    (waiting Confirmation, card, summary, each field, note)."""
    if isinstance(outcome, Confirmation):
        note = (
            "This needs your Confirmation: check it below, edit if needed, then approve or reject."
        )
        return [*history, {"role": "assistant", "content": note}], *confirmation_card(outcome)
    # Any charts in the answer show as images.
    return [*history, *charts.chat_messages(outcome)], *confirmation_card(None)


def confirmation_card(confirmation: Confirmation | None) -> list:
    """Outputs: waiting Confirmation, card, summary, each of CARD_FIELDS, note."""
    if confirmation is None:
        hidden = [gr.update(visible=False, value="") for _ in CARD_FIELDS]
        return [None, gr.update(visible=False), "", *hidden, ""]
    fields = confirmation.fields
    boxes = [
        gr.update(visible=name in fields, value=str(fields.get(name, ""))) for name in CARD_FIELDS
    ]
    summary = f"**Perry will do this only if you approve:**\n\n```\n{confirmation.summary}\n```"
    return [confirmation, gr.update(visible=True), summary, *boxes, ""]


def approve(
    request: gr.Request,
    thread_id: str | None,
    waiting: Confirmation | None,
    history: list[dict],
    *values,
):
    """Approve the waiting Write Action with the card's fields (edited or not)."""
    thread = thread_of(request, thread_id)
    if waiting is None:
        return history, *confirmation_card(None)
    edits = {
        name: value
        for name, value in zip(CARD_FIELDS, values, strict=True)
        if name in waiting.fields
    }
    return show(thread.approve(edits), history)


def reject(
    request: gr.Request,
    thread_id: str | None,
    waiting: Confirmation | None,
    history: list[dict],
    note: str,
):
    """Reject the waiting Write Action; the note goes to Perry to revise."""
    thread = thread_of(request, thread_id)
    if waiting is None:
        return history, *confirmation_card(None)
    shown = f"Rejected: {note}" if note.strip() else "Rejected."
    history = [*history, {"role": "user", "content": shown}]
    return show(thread.reject(note), history)


def hear(request: gr.Request, recording: str | None, history: list[dict], thread_id: str | None):
    """Push-to-talk: transcribe the recording and show it as the user's message.
    Outputs: message, chat, recorder, and what was Heard (None if nothing is sent)."""
    thread = thread_of(request, thread_id)
    if not recording:
        return gr.skip(), history, None, None
    try:
        heard = thread.hear(Path(recording).read_bytes())
    except usage.DailyCapReached as refused:
        return EMPTY, [*history, {"role": "assistant", "content": str(refused)}], None, None
    except Exception as error:  # noqa: BLE001 - tell the user, keep the UI usable
        gr.Warning(f"Couldn't transcribe the recording: {error}")
        return EMPTY, history, None, None
    if not heard.transcript.strip():
        gr.Warning("I didn't catch anything. Hold the button and speak, then release.")
        return EMPTY, history, None, None
    message = {"text": heard.transcript, "files": []}
    return message, [*history, {"role": "user", "content": heard.transcript}], None, heard


def read_aloud(
    request: gr.Request,
    heard: threads.Heard | None,
    history: list[dict],
    thread_id: str | None,
):
    """Speak the latest answer, only when the request came by voice (as `heard`)."""
    thread = thread_of(request, thread_id)
    if heard is None or not history or history[-1]["role"] != "assistant":
        return None
    text = _text(history[-1]["content"])
    if not text:
        return None
    try:
        speech = thread.read_aloud(heard, text)
    except Exception as error:  # noqa: BLE001 - the text answer is already shown
        gr.Warning(f"Couldn't read the answer aloud: {error}")
        return None
    if speech.note:
        gr.Warning(speech.note)
    return speech.audio


def show_if_spoken(audio):
    """Show the answer player only when there is an answer to play."""
    return gr.Audio(visible=audio is not None)


def _text(content) -> str:
    """A chat message's text: Gradio passes content as a str or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def open_thread(request: gr.Request, thread_id: str | None):
    """Show the Thread. Another User's Thread opens as if it didn't exist."""
    user = user_of(request)
    try:
        thread = threads.open(user, thread_id or None)
    except threads.ThreadNotFound:
        thread = threads.open(user, None)
    history = []
    for message in thread.history():
        is_answer = message["role"] == "assistant"
        history += charts.chat_messages(message["content"]) if is_answer else [message]
    pending = thread.pending_confirmation()
    return history, thread.id, *confirmation_card(pending), deletable(thread.id)


def new_thread():
    """Outputs: chat, Thread, Thread picker, card outputs, Delete thread button."""
    return [], None, gr.update(value=None), *confirmation_card(None), deletable(None)


def deletable(thread_id: str | None):
    """The Delete thread button: on only for a Thread that has been started."""
    return gr.update(interactive=thread_id is not None)


def ask_delete(request: gr.Request, thread_id: str | None):
    """Open the pop-up that asks before deleting the Thread. Outputs: pop-up, question,
    and the Thread it asks about (kept, so it is that one the Delete button deletes)."""
    thread = thread_of(request, thread_id)
    if thread.id is None:
        return gr.skip(), gr.skip(), None
    question = f'Delete "{thread.title()}"? Its messages can\'t be brought back.'
    return gr.update(visible=True), question, thread.id


def delete_thread(request: gr.Request, doomed: str | None):
    """Delete the Thread the pop-up asked about and show a new, empty one. Outputs:
    pop-up, asked-about Thread, then `new_thread()`'s outputs with the picker refreshed."""
    user = user_of(request)
    if doomed is not None:
        try:
            threads.delete(user, doomed)
        except threads.ThreadNotFound as missing:
            raise gr.Error(str(missing)) from None
    chat, thread_id, _, *rest = new_thread()
    picker = gr.update(choices=threads.list_recent(user), value=None)
    return gr.update(visible=False), None, chat, thread_id, picker, *rest


def cancel_delete():
    """Close the pop-up; nothing is deleted. Outputs: pop-up, asked-about Thread."""
    return gr.update(visible=False), None


def thread_choices(request: gr.Request):
    """The User's Thread picker list as it is now (the page may have been built long ago)."""
    return gr.update(choices=threads.list_recent(user_of(request)))


def lock_threads():
    """While a Turn runs, the New thread button, Thread picker and Delete thread button
    are off: a Turn that finished after a switch would otherwise write its Thread over
    the one being shown, or into one just deleted."""
    return gr.update(interactive=False), gr.update(interactive=False), deletable(None)


def unlock_threads(thread_id: str | None):
    return gr.update(interactive=True), gr.update(interactive=True), deletable(thread_id)


def propose_upload(
    request: gr.Request,
    files: list[str] | None,
    reader: str | None = None,
    chunking: str | None = None,
):
    """Show what ingesting the uploaded files would cost, read with `reader` and split
    with `chunking`, and wait for the Owner's OK to read them for Review."""
    owner_of(request)
    if not files:
        return gr.skip(), gr.skip(), gr.skip()
    estimate = ingest.estimate_ingestion([Path(file) for file in files], chunking, reader)
    try:
        ingest.check_budget(estimate)
    except ingest.OverBudget as refused:
        return [], str(refused), gr.update(visible=False)
    if estimate.unchanged == estimate.files:
        return [], "Already stored as it is, read and split the same way.", gr.update(visible=False)
    question = f"{estimate.describe()} Read for Review? Nothing is indexed until you approve."
    return files, question, gr.update(visible=True)


def read_upload(
    request: gr.Request, files: list[str], reader: str | None = None, chunking: str | None = None
):
    """Read and chunk the uploaded files for the Owner's Review; nothing is stored yet.
    Shows the Review panel at the first page read."""
    owner_of(request)
    try:
        read, left_out = ingest.read_for_review([Path(file) for file in files], chunking, reader)
    except ingest.OverBudget as refused:
        return [], str(refused), gr.update(visible=False), *review_panel([])
    lines = [f"- {result.describe()}" for result in left_out]
    if read:
        lines.append("Review the Chunks, then approve to index them, or discard them.")
    return read, "\n".join(lines), gr.update(visible=False), *review_panel(read)


def approve_review(request: gr.Request, read: list[ingest.ReadDocument]):
    """Index the reviewed Documents, each exactly as the Review showed it."""
    owner_of(request)
    if not read:
        return [], "Nothing to index.", *review_panel([])
    results = ingest.store_reviewed(read)
    # The Documents are searchable now; Summaries follow without holding up the upload.
    summaries.start_background()
    return [], "\n".join(f"- {r.describe()}" for r in results), *review_panel([])


def discard_review():
    return [], "Discarded: nothing indexed.", *review_panel([])


def cancel_upload():
    return [], "Nothing ingested.", gr.update(visible=False)


def review_panel(read: list[ingest.ReadDocument]) -> list:
    """The Review panel for `read`: shown at its first page, or hidden if there is none.
    In order: the panel, the summary, the page picker, the page image, the page's Chunks."""
    if not read:
        return [gr.update(visible=False), "", gr.update(choices=[], value=None), None, ""]
    choices = review_pages(read)
    first = choices[0][1]
    return [
        gr.update(visible=True),
        review_summary(read),
        gr.update(choices=choices, value=first),
        *review_page(read, first),
    ]


def review_summary(read: list[ingest.ReadDocument]) -> str:
    """A row per Document read: how it was read and split, and what came out."""
    rows = [
        (
            "| Document | Reader | Chunking Strategy | Pages | Chunks | Median characters "
            "| Most tokens | Pages not read | Fingerprint |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for document in read:
        lengths = sorted(len(chunk.text) for chunk in document.chunks)
        median = lengths[len(lengths) // 2] if lengths else 0
        tokens = max((models.count_embedding_tokens(c.text) for c in document.chunks), default=0)
        unread = ", ".join(map(str, document.untranscribed_pages)) or "none"
        rows.append(
            f"| {document.name} | {document.reader} | {document.chunking} "
            f"| {documents.page_count(document.path)} | {len(document.chunks)} | {median} "
            f"| {tokens} | {unread} | `{document.fingerprint}` |"
        )
    return "\n".join(rows)


def review_pages(read: list[ingest.ReadDocument]) -> list[tuple[str, str]]:
    """The page picker's choices: every page of every Document read, those with no
    Chunks too (a page that came out empty is what a Review is for)."""
    choices = []
    for index, document in enumerate(read):
        for page in range(1, documents.page_count(document.path) + 1):
            count = sum(1 for chunk in document.chunks if chunk.page == page)
            label = f"{document.name} · page {page} · {count} Chunks"
            if page in document.untranscribed_pages:
                label += " · not read"
            choices.append((label, f"{index}:{page}"))
    return choices


def review_page(read: list[ingest.ReadDocument], picked: str | None) -> list:
    """The page picked (`index:page`) as an image (PDFs only) beside its Chunks."""
    if not read or not picked:
        return [None, ""]
    index, page = (int(part) for part in picked.split(":"))
    document = read[index]
    image = None
    if document.path.suffix.lower() == ".pdf":
        image = PIL.Image.open(io.BytesIO(documents.render_page(document.path, page)))
    numbered = [(n, c) for n, c in enumerate(document.chunks, start=1) if c.page == page]
    parts = [f"<p><b>{html.escape(document.name)}, page {page}</b>: {len(numbered)} Chunks</p>"]
    if page in document.untranscribed_pages:
        parts.append("<p><b>This page couldn't be read: nothing on it would be searchable.</b></p>")
    for number, chunk in numbered:
        tokens = models.count_embedding_tokens(chunk.text)
        parts.append(
            f'<div class="review-chunk"><div class="review-chunk-meta">Chunk {number} of '
            f"{len(document.chunks)} · {len(chunk.text)} characters · {tokens} tokens</div>"
            f"<pre>{html.escape(chunk.text)}</pre></div>"
        )
    return [image, "\n".join(parts)]


def upload_datasets(request: gr.Request, files: list[str] | None) -> tuple[str, str]:
    owner_of(request)
    if not files:
        return gr.skip(), gr.skip()
    lines = []
    for file in files:
        try:
            lines += [result.describe() for result in datasets.upload(Path(file))]
        except Exception as error:  # noqa: BLE001 - report per file, keep going with the rest
            lines.append(f"{Path(file).name}: failed ({error})")
    # The Datasets are queryable now; the model describes what's left meanwhile.
    dataset_descriptions.start_background()
    return "\n".join(f"- {line}" for line in lines), dataset_list()


def dataset_list() -> str:
    found = datasets.list_datasets()
    if not found:
        return "_No Datasets yet._"
    return "\n".join(
        f"- **{d.name}** ({d.row_count} rows): {d.describe_for_panel()}" for d in found
    )


def summarising_status() -> str:
    """The sidebar's note on Documents still waiting for their Summary; empty if none.
    Without a local summary model nothing is being summarised, so it says what's needed."""
    pending = summaries.pending_count()
    if not pending:
        return ""
    if not models.summaries_supported():
        return models.summary_model_hint()
    return f"Summarising {pending} Document{'' if pending == 1 else 's'}…"


def todays_usage() -> str:
    today = usage.today()
    return f"Today: {today.tokens:,} tokens · ${today.cost_usd:.4f} of ${usage.daily_cap_usd():.2f}"


GOOGLE_BUTTON = {
    "connected": ("Google connected", "secondary"),
    "not connected": ("Connect Google", "primary"),
    "needs reconnecting": ("Reconnect Google", "stop"),
}


def google_button():
    """The Google button as the current Workspace status calls for."""
    label, variant = GOOGLE_BUTTON[workspace.current().status()]
    return gr.update(value=label, variant=variant)


ACCOUNT_KINDS = [("Test account (fake data only)", "test"), ("Real account", "real")]


def connect_google(request: gr.Request, kind: str | None):
    owner_of(request)
    if kind is None:
        return gr.skip(), "Choose whether this is a test or a real account first."
    try:
        google_auth.connect(kind)  # opens the browser and waits for consent
        message = f"Google connected ({kind} account)."
    except privacy.Refused as refusal:
        message = str(refusal)
    except Exception as error:  # noqa: BLE001 - show it next to the button, keep the UI up
        message = f"Couldn't connect Google: {error}"
    return google_button(), message


def sidebar_status():
    """Outputs: the Google button and today's usage, both as they are now."""
    return google_button(), todays_usage()


def build(privacy_notice: str = "") -> gr.Blocks:
    """The chat UI. `privacy_notice` (from `privacy.startup()`) shows under the Google button."""
    with gr.Blocks(title="Perry") as demo:
        thread_id = gr.State(None)
        pending = gr.State(EMPTY)
        heard = gr.State(None)  # the latest request, when it came by voice

        with gr.Sidebar():
            gr.Markdown("## Perry")
            who = gr.Markdown()
            gr.Button("Profile", link="/profile", size="sm")
            gr.Button("Log out", size="sm").click(None, js=LOG_OUT_JS)
            new_button = gr.Button("New thread", variant="primary")
            # Filled on page load with the User's own Threads.
            thread_picker = gr.Dropdown(choices=[], label="Threads", value=None)
            # Off until a Thread is open; it asks in a pop-up before deleting anything.
            delete_button = gr.Button("Delete thread", size="sm", interactive=False)
            gr.Markdown("### Documents")
            technique = gr.Dropdown(
                choices=technique_choices(),
                value=AUTO,
                label="Retrieval Technique",
                info="Auto lets Perry pick; choose one to force it for the next questions.",
            )
            # Owner-only controls start hidden; owner_controls shows them to the Owner.
            with gr.Column(visible=False) as document_controls:
                reader = gr.Dropdown(
                    choices=list(documents.READERS), value=documents.READER, label="Reader"
                )
                chunking = gr.Dropdown(
                    choices=list(documents.CHUNKINGS),
                    value=documents.CHUNKING,
                    label="Chunking Strategy",
                )
                uploader = gr.File(
                    file_count="multiple", file_types=[".pdf", ".md", ".txt"], label="Upload"
                )
                upload_status = gr.Markdown()
                with gr.Row(visible=False) as upload_choice:
                    read_button = gr.Button("Read for Review", variant="primary", size="sm")
                    cancel_button = gr.Button("Cancel", size="sm")
            summarising = gr.Markdown()
            # Summaries are written in the background, so the count is polled.
            summarising_timer = gr.Timer(10)
            uploaded = gr.State([])  # files waiting for the Owner's OK to ingest
            gr.Markdown("### Datasets")
            with gr.Column(visible=False) as dataset_controls:
                dataset_uploader = gr.File(
                    file_count="multiple", file_types=[".csv", ".xlsx"], label="Upload data"
                )
                dataset_status = gr.Markdown()
            datasets_shown = gr.Markdown(dataset_list)
            usage_status = gr.Markdown()
            with gr.Column(visible=False) as workspace_controls:
                gr.Markdown("### Calendar and email")
                account_kind = gr.Radio(ACCOUNT_KINDS, value=None, label="Account")
                google = gr.Button()
                google_status = gr.Markdown()

        # The pop-up that asks before a Thread is deleted (styled as one by CSS).
        doomed = gr.State(None)  # the Thread it asks about
        with (
            gr.Column(visible=False, elem_id="delete-thread-popup") as delete_popup,
            gr.Group(elem_id="delete-thread-dialog"),
        ):
            delete_question = gr.Markdown(padding=True)
            with gr.Row():
                confirm_delete_button = gr.Button("Delete", variant="stop")
                cancel_delete_button = gr.Button("Cancel", elem_id="delete-thread-cancel")
        # The Review: the Owner checks each page's Chunks before anything is indexed.
        reviewing = gr.State([])  # the Documents read, waiting for the Owner's approval
        with gr.Column(visible=False) as review:
            gr.Markdown("## Review before indexing")
            review_table = gr.Markdown()
            review_picker = gr.Dropdown(choices=[], label="Page", value=None)
            with gr.Row():
                review_image = gr.Image(
                    label="Page as read", type="pil", interactive=False, height="70vh"
                )
                review_chunks = gr.HTML(elem_id="review-chunks")
            with gr.Row():
                approve_review_button = gr.Button("Approve and index", variant="primary")
                discard_review_button = gr.Button("Discard", variant="stop")
        review_outputs = [review, review_table, review_picker, review_image, review_chunks]
        chatbot = gr.Chatbot(height="75vh", label="Perry")
        # The Confirmation card: one Write Action, shown exactly as it will be performed.
        waiting = gr.State(None)  # the Confirmation the card shows
        with gr.Group(visible=False) as card:
            card_summary = gr.Markdown()
            card_fields = [
                gr.Textbox(label=label, lines=lines, interactive=editable, visible=False)
                for label, lines, editable in CARD_FIELDS.values()
            ]
            card_note = gr.Textbox(label="Note for Perry if you reject (what to change)", lines=2)
            with gr.Row():
                approve_button = gr.Button("Approve", variant="primary")
                reject_button = gr.Button("Reject", variant="stop")
        card_outputs = [waiting, card, card_summary, *card_fields, card_note]
        # Images (screenshots, photos) can be dropped or attached alongside the text.
        box = gr.MultimodalTextbox(
            placeholder="Ask Perry…",
            show_label=False,
            submit_btn=True,
            file_types=["image"],
            file_count="multiple",
        )
        with gr.Row(visible=models.speech_supported()):
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                format="wav",
                label="Push to talk: record, then stop to send",
            )
            # Hidden until there is a spoken answer to play.
            speaker = gr.Audio(
                label="Perry's answer", autoplay=True, interactive=False, visible=False
            )

        status_outputs = [google, usage_status]

        def then_refresh_status(event):
            """After anything that may call a model or the Workspace, refresh the sidebar."""
            return event.then(sidebar_status, None, status_outputs)

        switching = [new_button, thread_picker, delete_button]

        def during_turn(event):
            """Lock Thread switching from `event` until the Turn's chain ends, then refresh
            the sidebar. `.then` (not `.success`) so a failed Turn still unlocks."""
            return then_refresh_status(event.then(unlock_threads, thread_id, switching))

        answer_outputs = [chatbot, thread_id, thread_picker, *card_outputs]
        during_turn(
            box.submit(lock_threads, None, switching)
            .then(show_user_message, [box, chatbot], [pending, chatbot, box])
            .then(answer, [pending, chatbot, thread_id, technique], answer_outputs)
        )
        box.submit(lambda: None, None, heard)
        during_turn(
            mic.stop_recording(lock_threads, None, switching)
            .then(hear, [mic, chatbot, thread_id], [pending, chatbot, mic, heard])
            .then(answer, [pending, chatbot, thread_id, technique, heard], answer_outputs)
            .then(read_aloud, [heard, chatbot, thread_id], speaker)
            .then(show_if_spoken, speaker, speaker)
        )
        during_turn(
            approve_button.click(lock_threads, None, switching).then(
                approve, [thread_id, waiting, chatbot, *card_fields], [chatbot, *card_outputs]
            )
        )
        during_turn(
            reject_button.click(lock_threads, None, switching).then(
                reject, [thread_id, waiting, chatbot, card_note], [chatbot, *card_outputs]
            )
        )
        thread_picker.input(
            open_thread, thread_picker, [chatbot, thread_id, *card_outputs, delete_button]
        )
        new_thread_outputs = [chatbot, thread_id, thread_picker, *card_outputs, delete_button]
        new_button.click(new_thread, None, new_thread_outputs)
        delete_button.click(ask_delete, thread_id, [delete_popup, delete_question, doomed]).then(
            None, js=FOCUS_CANCEL
        )
        confirm_delete_button.click(
            delete_thread, doomed, [delete_popup, doomed, *new_thread_outputs]
        )
        cancel_delete_button.click(cancel_delete, None, [delete_popup, doomed])
        for switch in (thread_picker.input, new_button.click, confirm_delete_button.click):
            switch(lambda: gr.Audio(value=None, visible=False), None, speaker)
        uploader.upload(
            propose_upload, [uploader, reader, chunking], [uploaded, upload_status, upload_choice]
        )
        then_refresh_status(
            read_button.click(
                read_upload,
                [uploaded, reader, chunking],
                [reviewing, upload_status, upload_choice, *review_outputs],
            )
        )
        review_picker.input(review_page, [reviewing, review_picker], [review_image, review_chunks])
        then_refresh_status(
            approve_review_button.click(
                approve_review, reviewing, [reviewing, upload_status, *review_outputs]
            ).then(summarising_status, None, summarising)
        )
        discard_review_button.click(
            discard_review, None, [reviewing, upload_status, *review_outputs]
        )
        cancel_button.click(cancel_upload, None, [uploaded, upload_status, upload_choice])
        dataset_uploader.upload(upload_datasets, dataset_uploader, [dataset_status, datasets_shown])
        demo.load(logged_in_as, None, who, api_name="logged_in_as")
        demo.load(
            owner_controls,
            None,
            [document_controls, dataset_controls, workspace_controls],
            api_name="owner_controls",
        )
        demo.load(sidebar_status, None, status_outputs)
        demo.load(summarising_status, None, summarising)
        summarising_timer.tick(summarising_status, None, summarising, show_progress="hidden")
        demo.load(thread_choices, None, thread_picker)
        demo.load(lambda: privacy_notice, None, google_status)
        google.click(connect_google, account_kind, [google, google_status])
    return demo


if __name__ == "__main__":
    # The chat UI is only served behind the login pages.
    from perry import web

    web.main()
