"""
Python REPL tool — executes arbitrary Python in a persistent namespace.

Key features:
- Persistent namespace across sequential calls (state accumulates)
- All agent tools exposed as plain async functions
- LLM callable as a simple async function: llm(prompt) -> str
- stdout + stderr captured and returned as the tool result
- One namespace per session_id
"""

import os
import asyncio
import contextvars
import functools
import io
import traceback
import warnings
from typing import Dict, Any, Optional

from ._tool_decorator import conditional_tool
from .context import get_session_context


# Per-session namespaces: session_id -> dict
_namespaces: Dict[str, Dict[str, Any]] = {}
_repl_process_lock: asyncio.Lock | None = None


def _get_repl_process_lock() -> asyncio.Lock:
    global _repl_process_lock
    if _repl_process_lock is None:
        _repl_process_lock = asyncio.Lock()
    return _repl_process_lock


class _ReplOutputLimitExceeded(Exception):
    pass


class _BoundedStringIO(io.StringIO):
    def __init__(self, max_chars: int, stream_name: str):
        super().__init__()
        self.max_chars = max_chars
        self.stream_name = stream_name

    def write(self, s: str) -> int:
        if self.max_chars > 0 and self.tell() + len(s) > self.max_chars:
            raise _ReplOutputLimitExceeded(
                f"{self.stream_name} exceeded REPL_OUTPUT_MAX_CHARS={self.max_chars}. "
                "The generated REPL code printed too much output. Rerun with a narrower query: "
                "print counts, filenames, matching line numbers, or small slices instead of full files/results."
            )
        return super().write(s)


def _get_or_create_namespace(session_id: str) -> Dict[str, Any]:
    """Get the persistent namespace for a session, creating it if needed."""
    if session_id not in _namespaces:
        _namespaces[session_id] = {"context": {}}
    ns = _namespaces[session_id]
    ns.setdefault("context", {})
    return ns


def inject_context_file(session_id: str, file_path: str, content: str) -> None:
    """Inject a file's content into the REPL namespace under context[file_path]."""
    ns = _get_or_create_namespace(session_id or "default")
    ns["context"][file_path] = content


def _captured_print_factory(stdout_capture: io.StringIO):
    def _captured_print(*args, sep=" ", end="\n", file=None, flush=False):
        target = stdout_capture if file is None else file
        text = sep.join(str(arg) for arg in args)
        target.write(text)
        target.write(end)
        if flush and hasattr(target, "flush"):
            target.flush()

    return _captured_print


def _write_captured_warnings(captured: list, stderr_capture: io.StringIO) -> None:
    for warning in captured:
        stderr_capture.write(
            warnings.formatwarning(
                str(warning.message),
                warning.category,
                warning.filename,
                warning.lineno,
                warning.line,
            )
        )


def _build_tool_callables(tools: list) -> Dict[str, Any]:
    """
    Wrap each BaseTool as a plain async callable.

    The REPL user calls: result = await read_file("src/main.py")
    Under the hood this calls tool.ainvoke({"file_path": "src/main.py"}).

    We also expose a sync shim so sync code works too.
    """
    callables = {}

    for tool in tools:
        name = tool.name

        # Capture tool in closure
        def make_async_fn(t):
            async def _async_tool(**kwargs):
                return await t.ainvoke(kwargs)
            _async_tool.__name__ = t.name
            _async_tool.__doc__ = t.description
            return _async_tool

        def make_sync_fn(t):
            def _sync_tool(**kwargs):
                """Sync shim — runs the async tool in the current event loop."""
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already inside an async context; schedule a task
                        import concurrent.futures
                        future = asyncio.run_coroutine_threadsafe(t.ainvoke(kwargs), loop)
                        return future.result(timeout=60)
                    else:
                        return loop.run_until_complete(t.ainvoke(kwargs))
                except Exception as e:
                    return f"Error calling {t.name}: {e}"
            _sync_tool.__name__ = t.name
            _sync_tool.__doc__ = t.description
            return _sync_tool

        callables[name] = make_async_fn(tool)
        # Also expose sync version with _sync suffix for convenience
        callables[f"{name}_sync"] = make_sync_fn(tool)

    return callables


def _build_llm_callable(provider, codebase_path: str = "", notes: str = "") -> Any:
    """
    Build a simple async llm(prompt) -> str callable.

    Usage in REPL:
        response = await llm("Summarize these findings: ...")
    """
    if provider is None:
        async def _no_llm(prompt: str, system: str = "") -> str:
            return "Error: LLM provider not initialized"
        return _no_llm

    async def _llm(prompt: str, system: str = "") -> str:
        """
        Call the LLM with a plain text prompt.

        Args:
            prompt: The user message
            system: Optional system message. If not provided, a codebase-aware
                    system prompt is injected automatically so the sub-LLM has
                    the same context (codebase path, current notes) as the main agent.

        Returns:
            The LLM's text response
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from .context import accumulate_repl_llm_tokens

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        else:
            # Inject codebase context so the sub-LLM can reason about project-specific content
            context_lines = ["You are a helpful assistant working inside a codebase analysis agent."]
            if codebase_path:
                context_lines.append(f"Codebase: {codebase_path}")
            if notes:
                context_lines.append(f"\nCurrent exploration notes:\n{notes}")
            context_lines.append("\nAnswer concisely and accurately based on the information provided.")
            messages.append(SystemMessage(content="\n".join(context_lines)))

        messages.append(HumanMessage(content=prompt))

        provider.bind_tools([])  # no tools for inline LLM calls
        response, _, usage = await provider.ainvoke(messages)

        accumulate_repl_llm_tokens(usage)

        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = "".join(
                part["text"] for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return content

    return _llm


@conditional_tool(True, parse_docstring=True)
async def python_repl(code: str) -> str:
    """
    Execute Python code in a persistent REPL session.

    The namespace persists across calls — variables, imports, and functions
    defined in one call are available in the next.
    
    An LLM is available as: response = await llm("your prompt here")

    OUTPUT DISCIPLINE — the REPL's stdout goes directly into context. Keep it small:
    - Never assign, print, or pass a full file or full tool result to anything. Filter immediately on the same line.
    - BAD:  content = await read_file(file_path="x.py"); print(content)
    - BAD:  content = await read_file(file_path="x.py"); result = await llm("summarise: " + content)
    - GOOD: lines = (await read_file(file_path="x.py")).splitlines(); print([l for l in lines if "def " in l])

    Returns stdout + stderr from the executed code.

    Args:
        code: Python code to execute
    """
    _, session_id = get_session_context()

    # Use "default" session if no session context set
    ns_key = session_id or "default"
    namespace = _get_or_create_namespace(ns_key)

    # Inject/refresh tool callables into namespace on every call
    # (uses the context tools list set at agent init)
    from .context import get_tools
    tool_callables = _build_tool_callables(get_tools())
    namespace.update(tool_callables)

    # Inject LLM callable — provider registered at agent init via set_llm_provider.
    # Pass codebase path and current notes so the sub-LLM has project context.
    from .context import get_llm_provider, get_codebase_path
    session_manager, _ = get_session_context()
    notes = session_manager.notes_for_prompt(session_id) if session_manager and session_id else ""
    codebase_path = str(get_codebase_path() or "")
    namespace["llm"] = _build_llm_callable(get_llm_provider(), codebase_path=codebase_path, notes=notes)

    # Inject standard useful builtins that aren't in exec's default namespace
    namespace.setdefault("asyncio", asyncio)

    # Per-execution timeout (seconds). Prevents infinite loops or stalled API
    # calls inside the REPL from hanging the entire eval.
    REPL_EXEC_TIMEOUT = int(os.getenv("REPL_EXEC_TIMEOUT", "60"))
    REPL_OUTPUT_MAX_CHARS = int(os.getenv("REPL_OUTPUT_MAX_CHARS", "100000"))

    # Capture stdout and stderr
    stdout_capture = _BoundedStringIO(REPL_OUTPUT_MAX_CHARS, "stdout")
    stderr_capture = _BoundedStringIO(REPL_OUTPUT_MAX_CHARS, "stderr")
    output_limit_errors: list[str] = []

    namespace["print"] = _captured_print_factory(stdout_capture)

    async with _get_repl_process_lock():
        old_cwd = os.getcwd()
        if codebase_path:
            os.chdir(codebase_path)
        try:
        # Detect top-level await before compiling — `await` outside a function is
        # a SyntaxError in normal exec mode, so we must wrap it first.
            import ast as _ast
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    tree = _ast.parse(code)
                has_await = any(
                    isinstance(node, (_ast.Await, _ast.AsyncFor, _ast.AsyncWith))
                    for node in _ast.walk(tree)
                )
            except SyntaxError:
                has_await = False

            if has_await:
                # Wrap in an async function so top-level await is valid.
                # Run directly on the main event loop — llm() uses an httpx AsyncClient
                # that is bound to this loop, so it must not be moved to a thread.
                indented = "\n".join(f"    {line}" for line in code.splitlines())
                async_code = f"async def __repl_async__():\n{indented}\n"
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    exec(compile(async_code, "<repl>", "exec"), namespace)
                    await asyncio.wait_for(namespace["__repl_async__"](), timeout=REPL_EXEC_TIMEOUT)
                    _write_captured_warnings(captured, stderr_capture)
            else:
                # Run synchronous exec in a thread so the event loop stays responsive,
                # while keeping normal print() output in the returned tool result.
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    compiled = compile(code, "<repl>", "exec")
                    _write_captured_warnings(captured, stderr_capture)

                def _exec_in_thread():
                    import traceback as _tb
                    try:
                        with warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter("always")
                            exec(compiled, namespace)
                            _write_captured_warnings(captured, stderr_capture)
                    except _ReplOutputLimitExceeded as exc:
                        output_limit_errors.append(str(exc))
                    except Exception:
                        stderr_capture.write(_tb.format_exc())

                loop = asyncio.get_running_loop()
                ctx = contextvars.copy_context()
                await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(ctx.run, _exec_in_thread)),
                    timeout=REPL_EXEC_TIMEOUT,
                )

        except _ReplOutputLimitExceeded as exc:
            output_limit_errors.append(str(exc))
        except asyncio.TimeoutError:
            stderr_capture.write(f"[REPL timeout] Execution exceeded {REPL_EXEC_TIMEOUT}s and was cancelled.\n")
        except Exception:
            stderr_capture.write(traceback.format_exc())
        finally:
            os.chdir(old_cwd)

    if output_limit_errors:
        return "[REPL output rejected]\n" + output_limit_errors[0]

    stdout_out = stdout_capture.getvalue()
    stderr_out = stderr_capture.getvalue()

    parts = []
    if stdout_out:
        parts.append(stdout_out.rstrip())
    if stderr_out:
        parts.append(f"[stderr]\n{stderr_out.rstrip()}")

    return "\n".join(parts) if parts else "(no output)"


def clear_repl_session(session_id: Optional[str] = None) -> None:
    """Clear the REPL namespace for a session (or all sessions if None)."""
    global _namespaces
    if session_id is None:
        _namespaces.clear()
    else:
        _namespaces.pop(session_id, None)
