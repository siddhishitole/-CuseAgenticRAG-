from __future__ import annotations
import typer
from rich.console import Console
from .router import run_agent, choose_agent
from .config import settings

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    question: str = typer.Argument(None, help="Your question to the agent."),
    agent: str = typer.Option(None, help="Force a specific agent: corrective"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive chat loop"),
):
    if interactive:
        return chat_loop()
    if not question:
        console.print("Please provide a question or use --interactive.")
        raise typer.Exit(code=1)
    if agent and agent not in {"corrective"}:
        console.print("Invalid agent. Choose from corrective.")
        raise typer.Exit(code=1)

    chosen = agent or choose_agent(question)
    console.print(f"[bold]Using agent:[/] {chosen}")
    answer = run_agent(question, chosen)
    console.print("\n[bold green]Answer:[/]\n" + answer)


def chat_loop():
    console.print("[bold]Interactive agenticRAG[/] (type 'exit' to quit)")
    while True:
        q = typer.prompt("You")
        if q.strip().lower() in {"exit", "quit"}:
            break
        chosen = choose_agent(q)
        console.print(f"[bold]CuseAgent:[/] {chosen}")
        try:
            a = run_agent(q, chosen)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
            continue
        console.print(f"[bold green]Answer:[/] {a}")


if __name__ == "__main__":
    app()
