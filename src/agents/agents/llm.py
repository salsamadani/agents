# coding=utf-8
# Copyright 2024 The AIWaves Inc. team.

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import time
from abc import abstractmethod
from typing import Union

import litellm
from dotenv import load_dotenv

from ..utils.config import Config
from ..utils.files import save_logs

load_dotenv()

WAIT_TIME = 20


def completion_with_backoff(**kwargs):
    """Call litellm.completion with retry on transient errors.

    Accepts ``api_key`` and ``api_base`` as keyword arguments so callers
    can pass provider-specific credentials without touching global state.
    """
    api_key = kwargs.pop("api_key", None)
    api_base = kwargs.pop("api_base", None)

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    while True:
        try:
            return litellm.completion(**kwargs)
        except litellm.OpenAIError:
            print(f"Please wait {WAIT_TIME} seconds and resend later ...")
            time.sleep(WAIT_TIME)


# ---------------------------------------------------------------------------
# Anthropic-format conversion helpers (used only by AnthropicBatchClient)
# ---------------------------------------------------------------------------

def _convert_messages_to_anthropic(messages):
    """Convert OpenAI-style messages to Anthropic format.

    Extracts system messages into a separate string and ensures alternating
    user/assistant roles as required by the Anthropic Messages API.

    Returns:
        tuple: (system_text, anthropic_messages)
    """
    system_parts = []
    non_system = []

    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            non_system.append(msg)

    system_text = "\n".join(system_parts) if system_parts else None

    merged = []
    for msg in non_system:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append({"role": msg["role"], "content": msg["content"]})

    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "Continue."})

    return system_text, merged


def _convert_tools_to_anthropic(openai_tools):
    """Convert OpenAI-style tool specs to Anthropic format."""
    anthropic_tools = []
    for tool in openai_tools:
        func = tool.get("function", {})
        anthropic_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools


def _convert_tool_choice_to_anthropic(tool_choice):
    """Convert OpenAI tool_choice to Anthropic format."""
    if tool_choice is None or tool_choice == "auto":
        return {"type": "auto"}
    elif tool_choice == "none":
        return None
    elif tool_choice == "required":
        return {"type": "any"}
    elif isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        return {"type": "tool", "name": func.get("name", "")}
    return {"type": "auto"}


def poll_batch_until_complete(batch_client, batch_id, poll_interval=60, timeout=86400):
    """Poll a Message Batch until processing completes.

    Args:
        batch_client: An ``AnthropicBatchClient`` instance.
        batch_id: The batch ID to poll.
        poll_interval: Seconds between polls (default 60).
        timeout: Maximum seconds to wait (default 86400 = 24h).

    Returns:
        The completed MessageBatch object.

    Raises:
        TimeoutError: If the batch does not complete within ``timeout`` seconds.
    """
    start = time.time()
    while True:
        batch = batch_client.retrieve_batch(batch_id)
        counts = batch.request_counts
        print(
            f"Batch {batch_id}: status={batch.processing_status} "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} "
            f"expired={counts.expired}"
        )
        if batch.processing_status == "ended":
            return batch
        elapsed = time.time() - start
        if elapsed + poll_interval > timeout:
            raise TimeoutError(
                f"Batch {batch_id} did not complete within {timeout}s"
            )
        time.sleep(poll_interval)


class LLMConfig(Config):

    required_fields = []

    def __init__(self, config_path_or_dict: Union[str, dict] = None):
        super().__init__(config_path_or_dict)
        self._validate_config()

        self.LLM_type: str = self.config_dict.get("LLM_type", "OpenAI")

        if self.LLM_type == "Anthropic":
            self.model: str = self.config_dict.get("model", "claude-sonnet-4-5")
            self.API_KEY: str = self.config_dict.get(
                "ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")
            )
            self.API_BASE = self.config_dict.get(
                "ANTHROPIC_BASE_URL", os.environ.get("ANTHROPIC_BASE_URL")
            )
        else:
            self.model: str = self.config_dict.get("model", "gpt-4-turbo-2024-04-09")
            self.API_KEY: str = self.config_dict.get(
                "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")
            )
            self.API_BASE = self.config_dict.get(
                "OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL")
            )

        self.temperature: float = self.config_dict.get("temperature", 0.3)
        self.log_path: str = self.config_dict.get("log_path", "logs")
        self.max_tokens: int = self.config_dict.get("max_tokens", 4096)
        self.MAX_CHAT_MESSAGES: int = self.config_dict.get("max_chat_messages", 10)
        self.ACTIVE_MODE: bool = self.config_dict.get("ACTIVE_MODE", False)
        self.SAVE_LOGS: bool = self.config_dict.get("SAVE_LOGS", False)


class LLM:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.temperature = self.config.temperature
        self.log_path = self.config.log_path
        self.API_KEY = self.config.API_KEY
        self.API_BASE = self.config.API_BASE
        self.max_tokens = self.config.max_tokens
        self.MAX_CHAT_MESSAGES = self.config.MAX_CHAT_MESSAGES
        self.ACTIVE_MODE = self.config.ACTIVE_MODE
        self.SAVE_LOGS = self.config.SAVE_LOGS

    @abstractmethod
    def get_response(cls, **kwargs):
        pass


class OpenAILLM(LLM):
    """LLM implementation using litellm for provider-agnostic completions.

    Despite the name, this class handles both OpenAI and Anthropic models
    via ``litellm``, which auto-detects the provider from the model name
    and normalizes responses into ``litellm.Message`` objects.
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    def get_stream(self, response, log_path, messages):
        ans = ""
        for res in response:
            if res:
                r = (
                    res.choices[0]["delta"].get("content")
                    if res.choices[0]["delta"].get("content")
                    else ""
                )
                ans += r
                yield r

        if self.SAVE_LOGS:
            save_logs(log_path, messages, ans)

    def get_response(
        self,
        chat_messages,
        system_prompt,
        last_prompt=None,
        stream=False,
        tools=None,
        tool_choice="auto",
        response_format=None,
        **kwargs,
    ):
        """
        return LLM's response
        """
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )

        if chat_messages:
            if len(chat_messages) > self.MAX_CHAT_MESSAGES:
                chat_messages = chat_messages[-self.MAX_CHAT_MESSAGES :]
            if isinstance(chat_messages[0], dict):
                messages += chat_messages

        if last_prompt:
            if self.ACTIVE_MODE:
                last_prompt += " Be accurate but concise in response."

            if messages:
                messages.append({"role": "user", "content": last_prompt})
            else:
                messages = [{"role": "user", "content": last_prompt}]

        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "api_key": self.API_KEY,
            "api_base": self.API_BASE,
        }

        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = tool_choice
        else:
            call_kwargs["stream"] = stream

        if response_format:
            call_kwargs["response_format"] = response_format

        response = completion_with_backoff(**call_kwargs)

        if stream:
            content = self.get_stream(response, self.log_path, messages)
        elif response.choices[0].message.get("tool_calls"):
            content = response.choices[0].message
        else:
            content = response.choices[0].message["content"].strip()
            if self.SAVE_LOGS:
                save_logs(self.log_path, messages, content)

        return response, content


# Keep backward-compatible alias
AnthropicLLM = OpenAILLM


class AnthropicBatchClient:
    """Client for the Anthropic Message Batches API (POST /v1/messages/batches).

    This is a standalone client that uses the ``anthropic`` SDK directly
    for batch operations, which are not supported by litellm.
    """

    def __init__(self, config: LLMConfig) -> None:
        import anthropic

        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.MAX_CHAT_MESSAGES = config.MAX_CHAT_MESSAGES
        self.ACTIVE_MODE = config.ACTIVE_MODE

        client_kwargs = {"api_key": config.API_KEY}
        if config.API_BASE:
            client_kwargs["base_url"] = config.API_BASE
        self.client = anthropic.Anthropic(**client_kwargs)

    def _build_messages(self, chat_messages, system_prompt, last_prompt):
        """Build OpenAI-style messages, then convert to Anthropic format."""
        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )

        if chat_messages:
            if len(chat_messages) > self.MAX_CHAT_MESSAGES:
                chat_messages = chat_messages[-self.MAX_CHAT_MESSAGES:]
            if isinstance(chat_messages[0], dict):
                messages += chat_messages

        if last_prompt:
            if self.ACTIVE_MODE:
                last_prompt += " Be accurate but concise in response."
            if messages:
                messages.append({"role": "user", "content": last_prompt})
            else:
                messages = [{"role": "user", "content": last_prompt}]

        return _convert_messages_to_anthropic(messages)

    def prepare_batch_request(
        self,
        custom_id,
        chat_messages,
        system_prompt,
        last_prompt=None,
        tools=None,
        tool_choice="auto",
        **kwargs,
    ):
        """Build a single request dict for the Message Batches API.

        Args:
            custom_id: Unique identifier for this request within the batch.
            chat_messages: Prior conversation messages.
            system_prompt: System prompt string.
            last_prompt: Final user prompt to append.
            tools: Optional OpenAI-style tool definitions.
            tool_choice: Tool choice strategy.
            **kwargs: Additional params (``max_tokens``, ``temperature``, ``model``).

        Returns:
            A dict with ``custom_id`` and ``params`` ready for
            ``create_batch()``.
        """
        system_text, anthropic_messages = self._build_messages(
            chat_messages, system_prompt, last_prompt
        )

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": "Hello."}]

        params = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_messages,
        }

        if system_text:
            params["system"] = system_text

        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            params["temperature"] = temperature

        if tools:
            params["tools"] = _convert_tools_to_anthropic(tools)
            converted_choice = _convert_tool_choice_to_anthropic(tool_choice)
            if converted_choice is not None:
                params["tool_choice"] = converted_choice

        return {"custom_id": custom_id, "params": params}

    def create_batch(self, requests):
        """Submit a Message Batch for asynchronous processing.

        Args:
            requests: List of batch request dicts from ``prepare_batch_request()``.

        Returns:
            A ``MessageBatch`` object with ``id``, ``processing_status``, etc.
        """
        return self.client.messages.batches.create(requests=requests)

    def retrieve_batch(self, batch_id):
        """Retrieve the current status of a Message Batch."""
        return self.client.messages.batches.retrieve(batch_id)

    def list_batches(self, limit=20):
        """List Message Batches, most recent first."""
        return self.client.messages.batches.list(limit=limit)

    def cancel_batch(self, batch_id):
        """Cancel a Message Batch that is still processing."""
        return self.client.messages.batches.cancel(batch_id)

    def get_batch_results(self, batch_id):
        """Retrieve results for a completed Message Batch.

        Returns:
            A list of result objects, each with ``custom_id`` and
            ``result`` (containing ``type`` and ``message`` on success).
        """
        return list(self.client.messages.batches.results(batch_id))


def create_llm(config: LLMConfig) -> LLM:
    """Factory function to create an LLM instance.

    Both OpenAI and Anthropic models are served by ``OpenAILLM`` via
    ``litellm``, which auto-detects the provider from the model name.
    """
    return OpenAILLM(config)
