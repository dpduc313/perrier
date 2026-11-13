# Any part of Perry can run on the company's model, one eval gate at a time

ADR-0006 put knowledge work on `inet-qwen` and kept the Supervisor, the Analysis Agent and the Task Agent off it, partly because its JSON choices only covered tools that take one `question`. The JSON choice now covers any tool: each step's JSON schema offers every bound tool with its own argument schema (taken from the tool, leaving out arguments Perry fills in itself, such as the run's config), or `answer`. The server's JSON mode keeps a choice within that schema, and Perry checks it again (an offered tool, its required arguments present) before turning it into the tool call; a choice that fails the check goes to the chat models like an unreachable server. One setting, `PERRY_INET_QWEN_FOR`, names what asks `inet-qwen` first: `knowledge` (the default: the Knowledge Agent, its Retrieval Techniques' calls and reading scanned pages), `techniques` (only those Retrieval Techniques' helper calls: ADR-0006's fallback), `analysis`, `task`, `supervisor`, or `all`. This reopens ADR-0006's rejected "`inet-qwen` drives every Agent and the Supervisor": its worry still stands, so the default stays knowledge work only and each other part moves only once the eval gate passes for it.

## Considered Options

- **A flat choice (`action`, free-form `arguments`) checked only by Perry**: rejected. With each tool's own schema in the JSON schema, the server can't return arguments of the wrong shape, and a live check answered tool choices with full arguments (an email's recipient, subject and body; a date range) in about a second.
- **Everything on `inet-qwen` at once**: rejected. Routing and Write Actions are where a weaker choice costs most, so each part gets its own eval comparison.
- **A setting per Agent (`PERRY_INET_QWEN_SUPERVISOR=true`, ...)**: rejected for one list, which is easier to read and to set for an eval run.

## Consequences

- The order we expect to move parts in: the Analysis Agent (Read Actions only, and its SQL is shown with the answer), then the Supervisor (routing), then the Task Agent (Write Actions). The Confirmation gate is middleware on the Write Action's tool call, so it holds whichever model proposes it.
- A real Google account still turns the Task Agent off: `inet-qwen` isn't local, and a step it can't answer goes to Gemini or OpenAI.
- The Supervisor reads the whole Thread, so a long Thread can pass `inet-qwen`'s 64k-token window; that call goes to the chat models.
- Speech stays on Gemini. Summaries and Document metadata stay on the local model (ADR-0003).
- Dataset descriptions are background work with no chat-model fallback, so they have their own setting, `PERRY_DATASET_DESCRIPTION_MODEL`, not `PERRY_INET_QWEN_FOR` (ADR-0008).
- Eval gate for a part: the eval set with that part added to `PERRY_INET_QWEN_FOR`, against the set without it, with the thresholds of ADR-0006.
