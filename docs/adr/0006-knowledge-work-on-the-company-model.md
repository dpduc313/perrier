# Knowledge work runs on the company's model, with tool calls as JSON choices

The Knowledge Agent, the helper calls its searches make (Question Decomposition, query rewrites, step-back, HyDE, self-query) and reading scanned pages run first on `inet-qwen`: a Qwen3.8-27B model the company serves with vLLM (`ai-api.inetcloud.vn`, model `knowledge`). The Supervisor, the Analysis Agent and the Task Agent stay on Gemini's free tier with the paid OpenAI fallback (ADR-0003). The server is for chat only and refuses tool calls, so the Knowledge Agent's steps ask it for a JSON choice instead (`{action: <an offered Retrieval Technique> | answer, question, answer}`, which the server's JSON mode always returns well-formed), and Perry turns that into the same tool call the loop has always made. We put knowledge work there because it is the bulk of a Turn's calls and the part about the Documents: on `inet-qwen` it costs nothing, isn't held to the free tier's few requests a minute, and keeps Document text inside the company. We kept routing and Write Actions off it because those choices matter most and a 27B model making tool calls in a format it wasn't tuned for may make them less well.

## Considered Options

- **`inet-qwen` drives every Agent and the Supervisor through JSON choices**: rejected. Routing between Agents and choosing Write Actions are the decisions we least want a smaller model making. (Reopened by ADR-0007: any part can move there, one eval gate at a time.)
- **The Knowledge Agent as fixed steps** (pick a technique, search, rephrase if nothing was found, write the findings) instead of a loop: rejected. The loop's one choice per step fits a single JSON schema, and keeping it leaves the Knowledge Agent working the same way on every model.
- **A list of tool calls per step**: rejected for now. One Retrieval Technique answers each question, so searches follow one another and a list would only add ways to go wrong. If evals show many questions spending their later searches on rephrasings, revisit it.
- **Asking for the server's tool calling to be turned on**: not available; the server is for chat only.

## Consequences

- Each step that fails on `inet-qwen` (unreachable, a server error, or slower than the chat timeout) falls back on its own to Gemini, then OpenAI, so one answer can mix models. The "Found with" line, when shown, names the model that answered.
- The server takes no API key and anyone who finds the URL can use it, so it can be busy; the timeout covers that. It has no request pacing, and Qwen's thinking stays off.
- `inet-qwen` is priced at $0, so its calls never count towards the Daily Cap; the Token Ceiling still stops a runaway Turn. Its context window is 64k tokens.
- Speech-to-text and read-aloud stay on Gemini: `inet-qwen` takes no audio.
- Benchmark and eval rows run knowledge work on `inet-qwen` and everything else on one fixed paid model, never the free tier.
- The switch-over gate: the eval set must score it close to the previous setup, with `gpt-5.4-mini` judging both. That means at most one fewer routing and Citation check passed, and a judge score within about 10%. Otherwise `inet-qwen` keeps only the helper calls.
- The gate ran on 2026-10-19 (#57), with `gpt-5.4-mini` answering the baseline, and failed by one check:
  - routing and Citation checks: 91/96 against 93/96;
  - judge score: 0.89 against 0.93 (−3.9%);
  - cost of the run: $0.16 against $0.25.
  All of the gap is one Turn. A comparing question was answered from one Document when it needed two.
- **We kept knowledge work on `inet-qwen` anyway.** It is the company's own model: we trust it with our Documents more than third-party APIs, and that outweighs one missed check.
- `PERRY_INET_QWEN_FOR=techniques` sends only the helper calls there, the fallback the gate called for, if answer quality ever matters more.
