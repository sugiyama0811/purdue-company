#!/usr/bin/env python3
"""
Purdue Research & Trading Company
Three-division CrewAI multi-agent system.

Usage:
    python main.py                                        # interactive mode
    python main.py "help me with heat transfer in fins"   # ME division
    python main.py "analyze ES futures"                   # trading division
    python main.py "find passive income ideas"            # passive income division
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env before any crewai imports
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table

from purdue_company.flow import CompanyFlow

console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗[/]
[bold cyan]║       PURDUE RESEARCH & TRADING COMPANY                     ║[/]
[bold cyan]║       Powered by CrewAI + Ollama (no API key required)      ║[/]
[bold cyan]╠══════════════════════════════════════════════════════════════╣[/]
[bold cyan]║  [green]ME Academic Division[/]    — coursework, research, design    ║[/]
[bold cyan]║  [yellow]Financial Trading[/]       — stocks & futures (ES/NQ)        ║[/]
[bold cyan]║  [magenta]Passive Income[/]          — AI-assisted income research      ║[/]
[bold cyan]╚══════════════════════════════════════════════════════════════╝[/]
"""

EXAMPLES = """[dim]
  [green]ME:[/green]      help me with Navier-Stokes in pipe flow
  [green]ME:[/green]      explain heat exchanger design for ME315
  [green]ME:[/green]      research papers on topology optimization

  [yellow]Trading:[/yellow] analyze ES futures
  [yellow]Trading:[/yellow] NQ day trading setup
  [yellow]Trading:[/yellow] what's the technical setup on NVDA

  [magenta]Passive:[/magenta] find passive income ideas for students
  [magenta]Passive:[/magenta] best dividend ETFs for F-1 visa holders
  [magenta]Passive:[/magenta] how to make money with AI tools passively

  Type [bold]quit[/bold] or [bold]q[/bold] to exit.
[/dim]"""


def validate_env():
    """Check that at least one LLM backend is configured."""
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_ollama = bool(os.getenv("OLLAMA_MODEL")) or True  # Ollama is always attempted

    if not has_anthropic:
        # Warn but don't exit — Ollama will be used
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code != 200:
                raise ConnectionError
        except Exception:
            console.print("\n[bold yellow]Ollama is not running.[/]")
            console.print("[yellow]Start it with:[/] [bold]ollama serve[/]")
            console.print("[yellow]Then pull a model:[/] [bold]ollama pull llama3.2[/]")
            console.print("\n[dim]Or set ANTHROPIC_API_KEY in .env to use Claude instead.[/]")
            sys.exit(1)


def show_division_label(division: str):
    labels = {
        "me": "[bold green]ME Academic Division[/]",
        "trading": "[bold yellow]Financial Trading Division[/]",
        "passive": "[bold magenta]Passive Income Division[/]",
        "unknown": "[bold red]Unknown[/]",
    }
    console.print(f"\nRouting to: {labels.get(division, division)}")


def run_request(user_input: str):
    console.print(f"\n[bold]Request:[/] {user_input}")
    console.print("[dim]Classifying and routing...[/]\n")

    flow = CompanyFlow()
    flow.state.user_request = user_input

    try:
        flow.kickoff()
        show_division_label(flow.state.division)

        output_path = flow.state.output_file
        if output_path and Path(output_path).exists():
            content = Path(output_path).read_text()
            border_colors = {"me": "green", "trading": "yellow", "passive": "magenta"}
            color = border_colors.get(flow.state.division, "cyan")
            console.print(
                Panel(
                    Markdown(content),
                    title=f"[bold]{output_path}[/]",
                    border_style=color,
                    padding=(1, 2),
                )
            )
            console.print(f"\n[dim]Report saved to:[/] [bold]{output_path}[/]")
        else:
            console.print(Panel(flow.state.final_output, border_style="cyan"))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        raise


def main():
    console.print(BANNER)
    validate_env()

    # Direct command mode
    if len(sys.argv) > 1:
        run_request(" ".join(sys.argv[1:]))
        return

    # Interactive mode
    console.print(EXAMPLES)

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]What would you like to do?[/]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                console.print("[cyan]Goodbye![/]")
                break
            run_request(user_input)
        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/]")
            break


if __name__ == "__main__":
    main()
