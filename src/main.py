#!/usr/bin/env python3
"""
Codebase Maintainer Agent CLI.
"""

from dotenv import load_dotenv; load_dotenv()
import asyncio, json, os, sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
console = Console()
print = console.print


from agent.maintainer import CodebaseAgent

class AgentCLI:
    """Command-line interface."""
    
    def __init__(self):
        self.agent = None
    
    async def initialize(self, codebase_path: str, session_id: str = None):
        """Initialize the agent."""
        print(f"\n🔍 Initializing agent for: {codebase_path}", style="bold blue")

        provider = os.getenv("LLM_PROVIDER", "anthropic")
        model = os.getenv("LLM_MODEL", None)
        if not model:
            raise ValueError("LLM_MODEL environment variable not set")

        self.agent = CodebaseAgent(
            codebase_path=codebase_path,
            provider=provider,
            model=model,
            session_id=session_id,
        )
        
        session_info = self.agent.get_session_info()
        print(f"✓ Session: {session_info['session_id']}", style="green")
        print(f"  Provider: {provider}", style="dim")
        print(f"  Model: {model if model else 'default'}", style="dim")
    
    async def chat_loop(self):
        """Main chat loop with streaming output."""
        print("\n💬 Chat with codebase (exit/quit/info)\n", style="bold")
        
        while True:
            try:
                question = Prompt.ask("[bold cyan]You[/bold cyan]")
                
                if not question.strip():
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n", style="bold green")
                    break
                
                if question.lower() == 'info':
                    self.show_session_info()
                    continue
                
                full_answer = ""
                citations = []
                final_tokens = None
                async for chunk in self.agent.ask_stream(question):
                    if chunk["type"] == "token":
                        pass

                    elif chunk["type"] == "tokens_usage":
                        node_tokens = chunk.get("tokens", {})
                        node = chunk.get("node", "unknown_node")
                        self._print_token_metrics(node_tokens, f"After {node}")

                    elif chunk["type"] == "tool_call":
                        args = chunk.get("args", {})
                        args_str = json.dumps(args, ensure_ascii=False)
                        args_preview = args_str if len(args_str) <= 120 else args_str[:120] + "…"
                        print(f"  [dim]→ {chunk['name']}({args_preview})[/dim]")

                    elif chunk["type"] == "tool_done":
                        result = chunk.get("result", "")
                        result_preview = result if len(result) <= 200 else result[:200] + "…"
                        print(f"  [dim]← {chunk['name']}: {result_preview}[/dim]")

                    elif chunk["type"] == "repl_code":
                        print(f"\n[bold yellow]  REPL ▶[/bold yellow]")
                        for line in chunk["content"].splitlines():
                            print(f"  [yellow]{line}[/yellow]")

                    elif chunk["type"] == "repl_output":
                        if chunk["content"] and chunk["content"] != "(no output)":
                            print(f"[bold green]  REPL ◀[/bold green]")
                            output = chunk["content"]
                            preview = output if len(output) <= 500 else output[:500] + "…"
                            for line in preview.splitlines():
                                print(f"  [green]{line}[/green]")

                    elif chunk["type"] == "done":
                        citations = chunk.get("citations", [])
                        full_answer = chunk.get("answer", "")
                        final_tokens = chunk.get("tokens_usage", None)
                
                print("\n[bold green]Assistant:[/bold green]")
                print(Markdown(full_answer))

                # Print citations
                if citations:
                    print("\n📚 References:", style="dim")
                    for citation in citations[:5]:
                        print(f"  • {citation}", style="dim")
                    
                # Print final token summary
                if final_tokens:
                    print("\n[bold cyan dim]═══ Workflow Token Summary ═══[/bold cyan dim]")
                    # tokens_usage is now a namespace breakdown; print each namespace then the total
                    if "total" in final_tokens:
                        for ns in ("orchestrator", "agent_builder", "subagents"):
                            bucket = final_tokens.get(ns, {})
                            if bucket.get("total_tokens", 0) > 0:
                                self._print_token_metrics(bucket, ns)
                        self._print_token_metrics(final_tokens["total"], "total")
                    else:
                        # fallback for old flat shape
                        self._print_token_metrics(final_tokens)

                print()  # Empty line for spacing
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n", style="bold green")
                break
            except Exception as e:
                console.print_exception(show_locals=False)

    def _print_token_metrics(self, token_usage: dict, node_context: str = ""):
        """
        Print styled token metrics to console.
        
        Shows input, output, and tool-contributed tokens with greyed styling for tool contribution.
        
        Args:
            token_usage: Dict with input_tokens, output_tokens, tool_output_contributed_input_tokens
            node_context: Context string (e.g., "After agent_step")
        """
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        tool_tokens = token_usage.get("tool_output_contributed_input_tokens", 0)
        cache_read = token_usage.get("cache_read_input_tokens", 0)
        cache_creation = token_usage.get("cache_creation_input_tokens", 0)

        # Format: "Input: 500 (150 from tool outputs) | Output: 200 | Total: 700"
        cache_parts = []
        if cache_read > 0:
            cache_parts.append(f"{cache_read} cache read")
        if cache_creation > 0:
            cache_parts.append(f"{cache_creation} cache write")
        if tool_tokens > 0:
            cache_parts.append(f"{tool_tokens} from tool outputs")

        if cache_parts:
            cache_note = f" [grey50 dim]({', '.join(cache_parts)})[/grey50 dim]"
        else:
            cache_note = ""

        metric_text = f"Input: {input_tokens}{cache_note} | Output: {output_tokens} | Total: {total_tokens}"
        
        # Print with context
        if node_context:
            print(f"[cyan dim]{node_context}[/cyan dim] • Tokens: {metric_text}")
        else:
            print(f"Tokens: {metric_text}")

    def show_session_info(self):
        """Display session info."""
        info = self.agent.get_session_info()
        text = (
            f"\nSession: {info['session_id']}\n"
            f"Codebase: {info['codebase_path']}\n"
            f"Created: {info['created_at']}\n"
            f"Conversations: {info['conversation_count']}\n"
        )
        print(Panel(text, title="Session Info"))

async def main():
    """Main entry point."""
    print("🚀 Starting Codebase Maintainer Agent CLI...\n")
    cli = AgentCLI()
    
    # Get codebase path and optional session ID
    if len(sys.argv) > 1:
        codebase_path = sys.argv[1]
    else:
        codebase_path = "."

    session_id = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize
    await cli.initialize(codebase_path, session_id=session_id)
    
    # Start chat loop
    await cli.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
