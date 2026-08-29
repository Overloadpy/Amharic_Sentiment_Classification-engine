"""Amharic Sentiment Classification Engine CLI and Test Harness.

Provides single-shot analysis, an interactive REPL, and automated golden benchmarks
using Typer and Rich.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.engine import SentimentInferenceEngine

app = typer.Typer(
    name="amharic-sentiment",
    help="Amharic Sentiment Classification & Verification CLI Harness",
    add_completion=False,
)
console = Console()

# Singleton engine instance for CLI operations
_ENGINE: SentimentInferenceEngine | None = None


def get_engine() -> SentimentInferenceEngine:
    """Retrieve or initialize the singleton SentimentInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = SentimentInferenceEngine()
        _ENGINE.load()
    return _ENGINE


def get_class_color(sentiment_class: str) -> str:
    """Map sentiment class to standard terminal colors."""
    mapping = {
        "Positive": "bold green",
        "Negative": "bold red",
        "Neutral": "bold yellow",
        "Mixed": "bold magenta",
    }
    return mapping.get(sentiment_class, "white")


@app.command()
def analyze(
    text: str = typer.Argument(..., help="Amharic text string to analyze"),
) -> None:
    """Classify sentiment of a single Amharic text input."""
    if not text.strip():
        console.print("[bold red]Error:[/bold red] Input text cannot be empty.")
        raise typer.Exit(code=1)

    engine = get_engine()
    result = engine.predict(text)

    color = get_class_color(result["class"])

    table = Table(title="Amharic Sentiment Analysis Result", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", width=18)
    table.add_column("Value", style="bold")

    table.add_row("Input Text", text)
    table.add_row("Cleaned Text", result["cleaned_text"])
    table.add_row("Predicted Class", f"[{color}]{result['class']}[/{color}]")
    table.add_row("Confidence", f"{result['confidence']:.2f}%")
    table.add_row("P(Positive)", f"{result['p_pos']:.4f}")
    table.add_row("P(Negative)", f"{result['p_neg']:.4f}")
    table.add_row("Latency", f"{result['latency_ms']:.2f} ms")

    console.print(table)


@app.command()
def repl() -> None:
    """Start an interactive Amharic sentiment REPL session."""
    console.print(
        Panel.fit(
            "[bold cyan]Amharic Sentiment Classification REPL[/bold cyan]\n"
            "Model: [green]Davlan/afro-xlmr-base[/green] | Dual-Axis Calibrated\n"
            "Type your Amharic sentence or type [yellow]:exit[/yellow] / [yellow]:quit[/yellow] to leave.",
            border_style="cyan",
        )
    )

    engine = get_engine()

    while True:
        try:
            user_input = console.input("[bold blue]amharic-nlp > [/bold blue]").strip()
            if not user_input:
                continue

            if user_input.lower() in [":exit", ":quit", "exit", "quit", ":q"]:
                console.print("[yellow]Exiting REPL session. Goodbye![/yellow]")
                break

            result = engine.predict(user_input)
            color = get_class_color(result["class"])

            console.print(
                f" -> Class: [{color}]{result['class']}[/{color}] | "
                f"Conf: [bold]{result['confidence']:.2f}%[/bold] | "
                f"P(pos): [green]{result['p_pos']:.3f}[/green] | "
                f"P(neg): [red]{result['p_neg']:.3f}[/red] | "
                f"Latency: [dim]{result['latency_ms']:.1f}ms[/dim]"
            )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break


@app.command()
def benchmark(
    cases_path: Path = typer.Option(
        Path("tests/benchmark_cases.json"),
        "--cases",
        "-c",
        help="Path to benchmark JSON cases",
    ),
) -> None:
    """Run evaluation over the Golden Benchmark Cases."""
    if not cases_path.exists():
        console.print(f"[bold red]Error:[/bold red] Benchmark cases file not found at {cases_path}")
        raise typer.Exit(code=1)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    console.print(f"[bold cyan]Running Amharic Sentiment Golden Benchmark ({len(cases)} cases)...[/bold cyan]\n")

    engine = get_engine()

    table = Table(title="Golden Benchmark Results", show_header=True, header_style="bold blue")
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Category", style="cyan", width=14)
    table.add_column("Text Preview", style="white", width=34)
    table.add_column("Expected", style="bold yellow", width=10)
    table.add_column("Predicted", width=12)
    table.add_column("Conf %", justify="right", width=8)
    table.add_column("P(pos)", justify="right", width=8)
    table.add_column("P(neg)", justify="right", width=8)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Status", justify="center", width=8)

    passed_count = 0
    total_latency = 0.0

    for item in cases:
        case_id = str(item.get("id", ""))
        category = item.get("category", "")
        text = item.get("text", "")
        expected = item.get("expected_class", "")

        res = engine.predict(text)
        predicted = res["class"]
        conf = res["confidence"]
        p_pos = res["p_pos"]
        p_neg = res["p_neg"]
        latency = res["latency_ms"]

        total_latency += latency
        is_pass = (predicted.lower() == expected.lower())
        if is_pass:
            passed_count += 1
            status_str = "[bold green]PASS[/bold green]"
        else:
            status_str = "[bold red]FAIL[/bold red]"

        pred_color = get_class_color(predicted)
        preview = (text[:30] + "..") if len(text) > 30 else text

        table.add_row(
            case_id,
            category,
            preview,
            expected,
            f"[{pred_color}]{predicted}[/{pred_color}]",
            f"{conf:.1f}%",
            f"{p_pos:.3f}",
            f"{p_neg:.3f}",
            f"{latency:.1f} ms",
            status_str,
        )

    console.print(table)

    accuracy = (passed_count / len(cases)) * 100.0 if cases else 0.0
    avg_latency = (total_latency / len(cases)) if cases else 0.0

    summary_panel = Panel.fit(
        f"[bold]Total Benchmark Cases:[/bold] {len(cases)}\n"
        f"[bold]Passed:[/bold] [green]{passed_count}[/green] / {len(cases)}\n"
        f"[bold]Accuracy:[/bold] [bold {'green' if accuracy >= 90 else 'yellow'}]{accuracy:.1f}%[/bold {'green' if accuracy >= 90 else 'yellow'}]\n"
        f"[bold]Avg Inference Latency:[/bold] {avg_latency:.2f} ms",
        title="[bold green]Benchmark Summary[/bold green]",
        border_style="green" if accuracy == 100 else "yellow",
    )
    console.print(summary_panel)


@app.command()
def gui() -> None:
    """Launch the modern Amharic Sentiment Intelligence Studio Desktop GUI."""
    try:
        from gui import main as launch_gui
        console.print("[bold green]Launching Amharic Sentiment Intelligence Studio...[/bold green]")
        launch_gui()
    except Exception as e:
        console.print(f"[bold red]Failed to launch GUI:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
