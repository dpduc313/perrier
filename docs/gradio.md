# Gradio 6 findings

What Perry's chat UI learned about Gradio 6 by reading `.venv/Lib/site-packages/gradio`. Check here before searching the package again, and add what you find.

## Pop-ups (modals)

Gradio 6 has no modal or dialog component. A pop-up is a hidden `gr.Column` with an `elem_id`, styled as an overlay in `chat_ui.CSS` (`position: fixed; inset: 0; z-index: 1000`), with a `gr.Group` inside as the dialog box. Show it by returning `gr.Column(visible=True)` from the event that asks; hide it from both answers. To move focus into it, chain `.then(None, js=...)` after the event that shows it.

Example: the delete-Thread pop-up in `perry/app.py` (`#delete-thread-popup`, `#delete-thread-dialog`, `FOCUS_CANCEL`).

## CSS and theme

`css=` is passed where the app is mounted (`gr.mount_gradio_app(..., css=chat_ui.CSS)` in `perry/web.py`), not to `gr.Blocks`.
