import ast
import html
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import rlm.clients as rlm_clients
import rlm.core.rlm as rlm_core
import rlm.utils.parsing as rlm_parsing
from rlm.clients.anthropic import AnthropicClient
from rlm.logger import RLMLogger
from rlm.environments.local_repl import LocalREPL

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent.sandbox import guarded  # noqa: E402

# One REPL action per root turn: the completion stops at the end of the first action, so the
# model sees each action's output before it writes the next one or its answer.
ONE_ACTION_PER_TURN = os.getenv("RLM_ONE_ACTION_PER_TURN") == "1"
ACTION_STOP = "</invoke>"
PROMPT_CACHE = os.getenv("RLM_PROMPT_CACHE") == "1"
FIRST_TURN_NOTICE = (
    "\n[runner] An answer submitted in the first turn is not accepted, because no REPL output "
    "had been seen yet. Inspect the output above, then submit."
)


ORIGINAL_FIND_CODE_BLOCKS = rlm_core.find_code_blocks
XML_TOOL_CALL_RE = re.compile(
    r"<invoke\s+name=[\"'](?P<name>repl|execute_code|run_python|python3|python)[\"'][^>]*>"
    r".*?"
    r"<parameter\b(?=[^>]*code)[^>]*>(?P<code>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
FENCED_REPL_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
XML_TAG_LINE_RE = re.compile(r"^\s*</?(?:repl|parameter|invoke|function_calls|antml)(?:\s+[^>]*)?>\s*$")


def clean_extracted_code(code: str) -> str:
    lines = html.unescape(code).strip().splitlines()
    kept = []
    for line in lines:
        if XML_TAG_LINE_RE.match(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


def find_code_blocks_with_xml_tools(response: str) -> list[str]:
    extracted = []

    for match in FENCED_REPL_RE.finditer(response):
        code = match.group(1).strip()
        cleaned = clean_extracted_code(code)
        if cleaned:
            extracted.append((match.start(), cleaned))

    for match in XML_TOOL_CALL_RE.finditer(response):
        code = match.group("code").strip()
        fenced = re.fullmatch(r"```(?:repl|python|py)?\s*\n(.*?)\n```", code, re.DOTALL)
        if fenced:
            code = fenced.group(1).strip()
        cleaned = clean_extracted_code(code)
        if cleaned:
            extracted.append((match.start(), cleaned))

    code_blocks = []
    for _, cleaned in sorted(extracted, key=lambda item: item[0]):
        if cleaned not in code_blocks:
            code_blocks.append(cleaned)

    final_files = None
    for match in reversed(list(re.finditer(r"FINAL\s*\(\s*(\[[\s\S]*?\])\s*\)", response))):
        try:
            files = ast.literal_eval(match.group(1))
        except Exception:
            continue
        if isinstance(files, list) and all(isinstance(path, str) for path in files):
            final_files = files
            break

    if final_files is not None:
        final_code = (
            "answer[\"content\"] = "
            + repr(final_files)
            + "\nanswer[\"ready\"] = True\nprint(\"FINAL\", answer[\"content\"])"
        )
        if final_code not in code_blocks:
            code_blocks.append(final_code)
    return code_blocks


def first_action_only(text: str) -> str:
    """Cut a reply after its first complete REPL action.

    The stop sequence does not always end the reply (the model sometimes writes actions in a
    function-call form that the API does not match against stop sequences), so the runner
    enforces one action per turn itself: later actions and any FINAL(...) after the first
    action are dropped before execution and are not kept in the conversation.
    """
    ends = [m.end() for m in (XML_TOOL_CALL_RE.search(text), FENCED_REPL_RE.search(text)) if m]
    if not ends:
        return text
    end = min(ends)
    closing = "\n</invoke>" if text[:end].rstrip().endswith("</parameter>") else ""
    return text[:end] + closing


def default_answer_with_user_prompt(self, message_history, lm_handler):
    current_prompt = message_history + [
        {
            "role": "user",
            "content": "Please provide a final answer to the user's question based on the information provided.",
        }
    ]
    try:
        response = lm_handler.completion(current_prompt)
    except Exception:
        response = self._best_partial_answer or ""
    if not response and self._best_partial_answer:
        response = self._best_partial_answer

    if self.logger:
        self.logger.log(
            rlm_core.RLMIteration(
                prompt=current_prompt,
                response=response,
                final_answer=response,
                code_blocks=[],
            )
        )

    return response


ORIGINAL_COMPLETION_TURN = rlm_core.RLM._completion_turn
ORIGINAL_COMPLETION = rlm_core.RLM.completion
ORIGINAL_EXECUTE_CODE = LocalREPL.execute_code


def completion_turn_observed_first(self, *args, **kwargs):
    """Refuse a final answer in the first root turn, before any REPL output was shown."""
    iteration = ORIGINAL_COMPLETION_TURN(self, *args, **kwargs)
    turn = getattr(self, "_root_turns_done", 0)
    self._root_turns_done = turn + 1
    if ONE_ACTION_PER_TURN and turn == 0:
        for block in iteration.code_blocks:
            result = block.result
            if getattr(result, "final_answer", None) is not None:
                result.final_answer = None
                result.stdout = (result.stdout or "") + FIRST_TURN_NOTICE
                self._first_turn_answers_refused = getattr(self, "_first_turn_answers_refused", 0) + 1
    return iteration


def completion_with_turn_count(self, *args, **kwargs):
    self._root_turns_done = 0
    self._first_turn_answers_refused = 0
    return ORIGINAL_COMPLETION(self, *args, **kwargs)


def execute_code_guarded(self, code):
    with guarded():
        return ORIGINAL_EXECUTE_CODE(self, code)


rlm_core.RLM._completion_turn = completion_turn_observed_first
rlm_core.RLM.completion = completion_with_turn_count
LocalREPL.execute_code = execute_code_guarded

rlm_parsing.find_code_blocks = find_code_blocks_with_xml_tools
rlm_core.find_code_blocks = find_code_blocks_with_xml_tools
rlm_core.RLM._default_answer = default_answer_with_user_prompt


class UsageAccumulator:
    def __init__(self):
        self.calls = defaultdict(int)
        self.input_tokens = defaultdict(int)
        self.output_tokens = defaultdict(int)
        self.cache_read = 0
        self.cache_write = 0

    def add(self, model, input_tokens, output_tokens, cache_read=0, cache_write=0):
        self.cache_read += cache_read
        self.cache_write += cache_write
        self.calls[model] += 1
        self.input_tokens[model] += input_tokens
        self.output_tokens[model] += output_tokens

    def to_model_summaries(self):
        return {
            model: {
                "total_calls": self.calls[model],
                "total_input_tokens": self.input_tokens[model],
                "total_output_tokens": self.output_tokens[model],
            }
            for model in sorted(self.calls)
        }

    def to_usage_summary(self):
        return {"model_usage_summaries": self.to_model_summaries()}


class TrackingAnthropicClient(AnthropicClient):
    def __init__(self, *args, usage_accumulator=None, request_retries=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_accumulator = usage_accumulator
        self.request_retries = request_retries

    @staticmethod
    def _is_retryable_error(exc) -> bool:
        text = str(exc).lower()
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 504, 529}:
            return True

        err_type = str(getattr(exc, "type", "")).lower()
        if err_type == "timeout_error":
            return True

        return (
            "timed out" in text
            or "timeout" in text
            or "interrupted" in text
            or "stream" in text and "closed" in text
        )

    @staticmethod
    def _finalize_stream(stream_obj):
        if hasattr(stream_obj, "__enter__"):
            with stream_obj as active_stream:
                return active_stream.get_final_message()
        return stream_obj.get_final_message()

    def _non_stream_request(self, kwargs):
        return self.client.messages.create(**kwargs)

    def _stream_request(self, kwargs):
        stream_factory = getattr(self.client.messages, "stream", None)
        if not callable(stream_factory):
            raise TypeError("messages.stream unavailable")

        response_kwargs = dict(kwargs)
        stream_obj = None
        try:
            stream_obj = stream_factory(**response_kwargs)
            response = self._finalize_stream(stream_obj)
            if response is None:
                raise RuntimeError("Streaming returned no final message")
            return response
        except TypeError as exc:
            # If SDK expects create(..., stream=True) while lacking .stream(),
            # fallback to create for this call path.
            if "argument" in str(exc).lower() and "stream" in str(exc).lower():
                response_kwargs["stream"] = True
                response = self.client.messages.create(**response_kwargs)
                if hasattr(response, "get_final_message"):
                    response = response.get_final_message()
                return response
            raise
        finally:
            if stream_obj is not None and not hasattr(stream_obj, "__enter__"):
                close = getattr(stream_obj, "close", None)
                if callable(close):
                    close()

    def completion(self, prompt, model=None):
        messages, system = self._prepare_messages(prompt)

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for Anthropic client.")

        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": messages}
        if PROMPT_CACHE:
            # Automatic prompt caching of the growing conversation: changes cost, not the model's input.
            kwargs["cache_control"] = {"type": "ephemeral"}
        if ONE_ACTION_PER_TURN:
            kwargs["stop_sequences"] = [ACTION_STOP]
        if system:
            kwargs["system"] = system

        max_attempts = max(1, int(self.request_retries) + 1)
        last_error = None

        for attempt in range(max_attempts):
            try:
                try:
                    response = self._stream_request(kwargs)
                except TypeError:
                    response = self._non_stream_request(kwargs)
            except Exception as exc:
                if attempt + 1 >= max_attempts or not self._is_retryable_error(exc):
                    raise
                last_error = exc
                time.sleep(min(2 ** attempt, 8))
                continue
            break

        if "response" not in locals():
            raise RuntimeError("RLM completion failed before receiving a response.") from last_error

        self._track_cost(response, model)
        text = "".join(
            getattr(block, "text", "")
            for block in (response.content or [])
            if getattr(block, "type", None) == "text"
        )
        return first_action_only(text) if ONE_ACTION_PER_TURN else text

    def _track_cost(self, response, model):
        # With prompt caching the API reports cached input separately; count it as input so that
        # token totals stay comparable with uncached runs, and keep the cache split for cost.
        u = response.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        total_in = u.input_tokens + cr + cw
        self.model_call_counts[model] += 1
        self.model_input_tokens[model] += total_in
        self.model_output_tokens[model] += u.output_tokens
        self.model_total_tokens[model] += total_in + u.output_tokens
        self.last_prompt_tokens = total_in
        self.last_completion_tokens = u.output_tokens
        if self.usage_accumulator is not None:
            self.usage_accumulator.add(model, total_in, u.output_tokens, cr, cw)


def drop_repl_locals(payload):
    iterations = payload.get("iterations") if isinstance(payload, dict) else None
    for iteration in iterations or ([payload] if isinstance(payload, dict) else []):
        for code_block in iteration.get("code_blocks", []) or []:
            result = code_block.get("result")
            if not isinstance(result, dict):
                continue
            result.pop("locals", None)
            for call in result.get("rlm_calls", []) or []:
                metadata = call.get("metadata")
                if isinstance(metadata, dict):
                    drop_repl_locals(metadata)
    return payload


class CompactRLMLogger(RLMLogger):
    def log(self, iteration):
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        drop_repl_locals(entry)
        self._iterations.append(entry)

        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")


@contextmanager
def capture_rlm_usage(accumulator):
    original_clients_get_client = rlm_clients.get_client
    original_core_get_client = rlm_core.get_client

    def tracked_get_client(backend, backend_kwargs):
        if backend == "anthropic":
            return TrackingAnthropicClient(
                usage_accumulator=accumulator,
                **(backend_kwargs or {}),
            )
        return original_clients_get_client(backend, backend_kwargs)

    rlm_clients.get_client = tracked_get_client
    rlm_core.get_client = tracked_get_client
    try:
        yield
    finally:
        rlm_clients.get_client = original_clients_get_client
        rlm_core.get_client = original_core_get_client
