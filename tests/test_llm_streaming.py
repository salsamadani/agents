import json
import unittest
from unittest.mock import patch

import httpx
import litellm
import requests

from agents.agents.llm import LLMConfig, create_llm


MODELS = {"OpenAI": "gpt-4-turbo-2024-04-09", "Anthropic": "anthropic/claude-sonnet-4-5"}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
    }
]
CALL = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"x":2}'}}


def openai_reply(body, tool_reply):
    header = {"id": "chatcmpl-test", "model": body["model"], "created": 1}
    finish = "tool_calls" if tool_reply else "stop"
    if body.get("stream"):
        if tool_reply:
            deltas = [
                {
                    "tool_calls": [
                        {**CALL, "index": 0, "function": {"name": "lookup", "arguments": '{"x":'}}
                    ]
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": "2}"}}]},
            ]
        else:
            deltas = [{"content": "hel"}, {"content": "lo"}]
        events = []
        for delta in [*deltas, {}]:
            choice = {"index": 0, "delta": delta, "finish_reason": None if delta else finish}
            chunk = {**header, "object": "chat.completion.chunk", "choices": [choice]}
            events.append("data: " + json.dumps(chunk))
        return "text/event-stream", "\n\n".join(events + ["data: [DONE]", ""]).encode()

    message = {"role": "assistant", "content": None if tool_reply else "hello"}
    if tool_reply:
        message["tool_calls"] = [CALL]
    choice = {"index": 0, "message": message, "finish_reason": finish}
    reply = {
        **header,
        "object": "chat.completion",
        "choices": [choice],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    return "application/json", json.dumps(reply).encode()


def anthropic_reply(body, tool_reply):
    header = {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "stop_sequence": None,
    }
    block = (
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"x": 2}}
        if tool_reply
        else {"type": "text", "text": "hello"}
    )
    finish = "tool_use" if tool_reply else "end_turn"
    if body.get("stream"):
        start = {**block, "input": {}} if tool_reply else {"type": "text", "text": ""}
        deltas = (
            [{"type": "input_json_delta", "partial_json": part} for part in ('{"x":', "2}")]
            if tool_reply
            else [{"type": "text_delta", "text": part} for part in ("hel", "lo")]
        )
        events = [
            {"type": "message_start", "message": {**header, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": start},
            *[{"type": "content_block_delta", "index": 0, "delta": delta} for delta in deltas],
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": finish, "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
        data = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return "text/event-stream", data.encode()
    return "application/json", json.dumps(
        {**header, "content": [block], "stop_reason": finish}
    ).encode()


class TestLLMStreaming(unittest.TestCase):
    def call_model(self, provider, *, stream, tools, tool_reply):
        bodies = []

        def openai_send(client, request, **kwargs):
            body = json.loads(request.content)
            bodies.append(body)
            media, data = openai_reply(body, tool_reply)
            return httpx.Response(
                200, headers={"content-type": media}, content=data, request=request
            )

        def anthropic_send(session, request, **kwargs):
            body = json.loads(request.body)
            bodies.append(body)
            media, data = anthropic_reply(body, tool_reply)
            response = requests.Response()
            response.status_code = 200
            response.headers["content-type"] = media
            response._content = data
            response._content_consumed = True
            return response

        key = provider.upper() + "_API_KEY"
        base = provider.upper() + "_BASE_URL"
        url = (
            "https://fixture.invalid/v1"
            if provider == "OpenAI"
            else "https://fixture.invalid/v1/messages"
        )
        client = create_llm(
            LLMConfig({"LLM_type": provider, "model": MODELS[provider], key: "fixture", base: url})
        )
        with (
            patch.object(httpx.Client, "send", openai_send),
            patch.object(requests.Session, "send", anthropic_send),
        ):
            result = client.get_response([], "", "fixture", stream=stream, tools=tools)
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bool(bodies[0].get("tools")), bool(tools))
        return result

    def test_stream_with_tools_exposes_structured_call(self):
        for provider in MODELS:
            with self.subTest(provider=provider):
                raw, _ = self.call_model(provider, stream=True, tools=TOOLS, tool_reply=True)
                calls = [
                    call for chunk in raw for call in (chunk.choices[0].delta.tool_calls or [])
                ]
                self.assertEqual({call.index for call in calls}, {0})
                self.assertEqual([call.id for call in calls if call.id], ["call_1"])
                self.assertEqual(
                    [call.function.name for call in calls if call.function.name], ["lookup"]
                )
                arguments = "".join(call.function.arguments or "" for call in calls)
                self.assertEqual(json.loads(arguments), {"x": 2})

    def test_streamed_text(self):
        for provider in MODELS:
            for tools in (TOOLS, None, []):
                with self.subTest(provider=provider, tools=tools):
                    _, content = self.call_model(
                        provider, stream=True, tools=tools, tool_reply=False
                    )
                    self.assertIs(iter(content), content)
                    self.assertEqual("".join(content), "hello")

    def test_nonstream_tool_call_returns_complete_message(self):
        for provider in MODELS:
            with self.subTest(provider=provider):
                raw, content = self.call_model(provider, stream=False, tools=TOOLS, tool_reply=True)
                self.assertIs(content, raw.choices[0].message)
                self.assertIsInstance(content, litellm.Message)
                (call,) = content.tool_calls
                self.assertEqual((call.id, call.function.name), ("call_1", "lookup"))
                self.assertEqual(json.loads(call.function.arguments), {"x": 2})
