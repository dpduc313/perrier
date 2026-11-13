"""Tool calls for a model that can't make them, as JSON choices (docs/adr/0006, 0007).

The company's model is served for chat only and refuses tool calls, but follows a JSON
schema. So each step asks it for one JSON object choosing what to do next (one of the
offered tools with its arguments, or the answer), constrained by a JSON schema built from
the tools' own, and the choice is turned into the tool call or answer the caller expects.
Any tool works, whatever arguments it takes.

Nothing here knows a provider: `perry.models` decides which model gets JSON choices.
"""

import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

_INSTRUCTIONS = """

You can't call tools directly: reply to every message with one JSON object, \
{{"step": ...}}, choosing your next step.
- To use a tool, set "action" to its name and "arguments" to what it takes.
- To finish, set "action" to "answer" and "answer" to your reply.

The tools (arguments marked ? may be left out):
{tools}"""


class InvalidChoice(ValueError):
    """The model replied with something that isn't one of the offered choices."""


def response_format(tools: list) -> dict:
    """The JSON schema a choice must follow: one of the offered tools with arguments its
    own schema allows, or "answer"."""
    steps = [
        {
            "type": "object",
            "properties": {
                "action": {"enum": [name]},
                "arguments": {"additionalProperties": False, **parameters, "type": "object"},
            },
            "required": ["action", "arguments"],
            "additionalProperties": False,
        }
        for name, parameters in _parameters(tools).items()
    ]
    steps.append(
        {
            "type": "object",
            "properties": {"action": {"enum": ["answer"]}, "answer": {"type": "string"}},
            "required": ["action", "answer"],
            "additionalProperties": False,
        }
    )
    schema = {
        "type": "object",
        "properties": {"step": {"anyOf": steps}},
        "required": ["step"],
        "additionalProperties": False,
    }
    return {"type": "json_schema", "json_schema": {"name": "step", "schema": schema}}


def conversation(messages: list, tools: list) -> list:
    """The conversation as a model without tool calls reads it: the tools described in
    the system prompt, each tool call as the JSON choice that made it, and each tool's
    result as a user message (such a server takes neither tool calls nor tool results)."""
    listed = "\n".join(
        f"- {_signature(function)}: {function.get('description', '')}"
        for function in map(_function, tools)
    )
    instructions = _INSTRUCTIONS.format(tools=listed)
    converted: list = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append(SystemMessage(message.text + instructions))
        elif isinstance(message, AIMessage) and message.tool_calls:
            converted.extend(AIMessage(_choice(call)) for call in message.tool_calls)
        elif isinstance(message, ToolMessage):
            converted.append(HumanMessage(f"Result of {message.name}:\n{message.text}"))
        else:
            converted.append(message)
    if not any(isinstance(message, SystemMessage) for message in converted):
        converted.insert(0, SystemMessage(instructions.lstrip()))
    return converted


def parse(reply: AIMessage, tools: list, messages: list) -> AIMessage:
    """The tool call or answer the JSON choice in `reply` stands for, keeping the reply's
    token counts and metadata. Raises InvalidChoice if it isn't one of the choices."""
    try:
        choice = json.loads(reply.text)
    except json.JSONDecodeError as error:
        raise InvalidChoice(f"Not a JSON choice: {reply.text[:200]!r}") from error
    step = choice.get("step") if isinstance(choice, dict) else None
    if not isinstance(step, dict):
        raise InvalidChoice(f"Not a JSON choice: {reply.text[:200]!r}")
    action = step.get("action")
    offered = _parameters(tools)
    if action == "answer":
        answer = (step.get("answer") or "").strip()
        if not answer:
            raise InvalidChoice("Chose to answer, with no answer.")
        message = AIMessage(answer)
    elif action in offered:
        arguments = step.get("arguments")
        if not isinstance(arguments, dict):
            raise InvalidChoice(f"Chose {action!r} without its arguments.")
        parameters = offered[action]
        # Without a question of its own, a search is for what the user asked.
        if (
            "question" in parameters.get("properties", {})
            and not str(arguments.get("question") or "").strip()
        ):
            arguments["question"] = next(
                (m.text for m in reversed(messages) if isinstance(m, HumanMessage)), ""
            )
        if missing := set(parameters.get("required", [])) - arguments.keys():
            raise InvalidChoice(f"Chose {action!r} without {', '.join(sorted(missing))}.")
        tool_call = {"name": action, "args": arguments, "id": f"call_{uuid.uuid4().hex}"}
        message = AIMessage("", tool_calls=[tool_call])
    else:
        raise InvalidChoice(f"Chose {action!r}, which isn't offered.")
    message.usage_metadata = reply.usage_metadata
    message.response_metadata = reply.response_metadata
    return message


def _choice(tool_call: dict) -> str:
    """The JSON choice that would have made `tool_call`."""
    return json.dumps({"step": {"action": tool_call["name"], "arguments": tool_call["args"]}})


def _function(tool) -> dict:
    """A tool's name, description and argument schema (arguments Perry fills in itself,
    such as the run's config, aren't the model's to choose)."""
    return convert_to_openai_tool(tool)["function"]


def _parameters(tools: list) -> dict[str, dict]:
    """Each tool's name and the JSON schema of the arguments it takes."""
    return {
        function["name"]: function.get("parameters") or {"type": "object", "properties": {}}
        for function in map(_function, tools)
    }


def _signature(function: dict) -> str:
    """A tool as the instructions list it, e.g. "run_python(code: string, inputs?: object)"."""
    parameters = function.get("parameters") or {}
    required = set(parameters.get("required", []))
    arguments = []
    for argument, schema in parameters.get("properties", {}).items():
        types = [s.get("type") for s in schema.get("anyOf", [schema])]
        kind = " or ".join(t for t in types if t and t != "null") or "any"
        arguments.append(f"{argument}{'' if argument in required else '?'}: {kind}")
    return f"{function['name']}({', '.join(arguments)})"
