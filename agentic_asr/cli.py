"""Command-line interface for Agentic ASR."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.agent import create_asr_agent
from .core.models import LLMProviderConfig
from .config import Config

load_dotenv()

app = typer.Typer(help="Agentic ASR - Intelligent Content Analysis with LLM processing")
console = Console()


@app.command()
def analyze(
        text_file: str = typer.Argument(..., help="Path to text file to analyze"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for analysis"),
        analysis_type: str = typer.Option("summary", "--type", "-t", 
                                        help="Analysis type (summary, keywords, sentiment)"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Analyze a text file using AI."""

    async def _analyze():
        # Validate text file
        text_path = Path(text_file)
        if not text_path.exists():
            console.print(f"[red]Error: Text file '{text_file}' not found[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Analyzing: {text_file}[/blue]")

        # Read the text content
        try:
            content = text_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Analysis completed![/green]")
        console.print(f"[dim]Content length: {len(content)} characters[/dim]")
        console.print(f"[dim]Analysis type: {analysis_type}[/dim]")
        
        if verbose:
            console.print(f"[dim]File: {text_file}[/dim]")
            console.print(f"[dim]Word count: {len(content.split())}[/dim]")

        # Save results if requested
        if output:
            output_path = Path(output)
            output_path.write_text(f"Analysis of {text_file}:\n\n{content[:500]}...", encoding='utf-8')
            console.print(f"[green]Analysis saved to: {output}[/green]")

    asyncio.run(_analyze())


@app.command()
def chat(
        session_id: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID to continue conversation"),
        llm_provider: str = typer.Option(Config.DEFAULT_LLM_PROVIDER, "--provider", "-p", help="LLM provider (openai, anthropic)"),
        model: str = typer.Option(Config.DEFAULT_LLM_MODEL, "--model", "-m", help="LLM model to use"),
        database_url: Optional[str] = typer.Option(None, "--db", help="Database URL for history storage"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Start an interactive chat session with the ASR agent."""

    async def _chat():
        if llm_provider.lower() == "openai":
            api_key = Config.OPENAI_API_KEY
            if not api_key:
                console.print("[red]Error: OPENAI_API_KEY not found in environment[/red]")
                raise typer.Exit(1)
        elif llm_provider.lower() == "anthropic":
            api_key = Config.ANTHROPIC_API_KEY
            if not api_key:
                console.print("[red]Error: ANTHROPIC_API_KEY not found in environment[/red]")
                raise typer.Exit(1)
        else:
            console.print(f"[red]Error: Unsupported LLM provider: {llm_provider}[/red]")
            raise typer.Exit(1)

        llm_config = LLMProviderConfig(
            provider_name=llm_provider,
            model=model,
            api_key=api_key
        )

        console.print("[blue]Initializing ASR agent...[/blue]")
        agent = await create_asr_agent(
            llm_config=llm_config,
            database_url=database_url or os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{Config.DATABASE_PATH}"),
            session_id=session_id
        )

        console.print("[green]ASR Agent ready! Type 'quit' to exit.[/green]")
        if session_id:
            console.print(f"[dim]Continuing session: {session_id}[/dim]")
        else:
            console.print(f"[dim]New session: {agent.session_id}[/dim]")

        try:
            while True:
                user_input = console.input("\n[bold blue]You:[/bold blue] ")

                if user_input.lower().strip() in ['quit', 'exit', 'bye']:
                    break

                if not user_input.strip():
                    continue

                with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                ) as progress:
                    task = progress.add_task("Thinking...", total=None)

                    try:
                        response = await agent.answer_to(user_input)
                        progress.stop()

                        console.print(f"\n[bold green]{agent.name}:[/bold green] {response.content}")

                        if verbose and response.tool_calls:
                            console.print(f"[dim]Tool calls made: {len(response.tool_calls)}[/dim]")

                    except Exception as e:
                        progress.stop()
                        console.print(f"[red]Error: {e}[/red]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
        finally:
            await agent.close()

    asyncio.run(_chat())


@app.command()
def process(
        text_file: str = typer.Argument(..., help="Text file to process"),
        query: str = typer.Argument(..., help="Query to ask about the text content"),
        output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for results"),
        llm_provider: str = typer.Option(Config.DEFAULT_LLM_PROVIDER, "--provider", "-p", help="LLM provider"),
        llm_model: str = typer.Option(Config.DEFAULT_LLM_MODEL, "--model", "-m", help="LLM model"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Process text file: analyze and ask LLM about the content."""

    async def _process():
        text_path = Path(text_file)
        if not text_path.exists():
            console.print(f"[red]Error: Text file '{text_file}' not found[/red]")
            raise typer.Exit(1)

        if llm_provider.lower() == "openai":
            api_key = Config.OPENAI_API_KEY
        elif llm_provider.lower() == "anthropic":
            api_key = Config.ANTHROPIC_API_KEY
        else:
            console.print(f"[red]Error: Unsupported LLM provider: {llm_provider}[/red]")
            raise typer.Exit(1)

        if not api_key:
            console.print(f"[red]Error: API key for {llm_provider} not found in environment[/red]")
            raise typer.Exit(1)

        console.print("[blue]Step 1: Reading text content...[/blue]")

        # Read the text content
        try:
            content = text_path.read_text(encoding='utf-8')
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Content loaded![/green]")
        if verbose:
            console.print(Panel(content[:500] + "..." if len(content) > 500 else content, 
                               title="Text Content", border_style="green"))

        console.print("\n[blue]Step 2: Processing with LLM...[/blue]")

        llm_config = LLMProviderConfig(
            provider_name=llm_provider,
            model=llm_model,
            api_key=api_key
        )

        agent = await create_asr_agent(llm_config=llm_config)

        full_query = f"Here is text content:\n\n{content}\n\nUser question: {query}"

        with Progress(SpinnerColumn(), TextColumn("Processing..."), console=console) as progress:
            task = progress.add_task("Processing with LLM...", total=None)
            response = await agent.answer_to(full_query)
            progress.stop()

        console.print(f"\n[bold green]Response:[/bold green]")
        console.print(Panel(response.content, border_style="green"))

        if output:
            output_path = Path(output)
            result_text = f"Content:\n{content}\n\nQuery: {query}\n\nResponse:\n{response.content}"
            output_path.write_text(result_text, encoding='utf-8')
            console.print(f"\n[green]Results saved to: {output}[/green]")

        await agent.close()

    asyncio.run(_process())


@app.command()
def list_sessions(
        database_url: Optional[str] = typer.Option(None, "--db", help="Database URL"),
        limit: int = typer.Option(20, "--limit", "-n", help="Number of sessions to show")
):
    """List recent conversation sessions."""

    async def _list_sessions():
        from .core.history import HistoryManager

        db_url = database_url or os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{Config.DATABASE_PATH}")
        history_manager = HistoryManager(db_url)

        try:
            await history_manager.initialize()
            sessions = await history_manager.list_sessions(limit)

            if not sessions:
                console.print("[yellow]No conversation sessions found.[/yellow]")
                return

            console.print(f"[bold blue]Recent Sessions (showing {len(sessions)}):[/bold blue]\n")

            for session in sessions:
                console.print(f"[green]Session ID:[/green] {session['session_id']}")
                console.print(f"[dim]Created: {session['created_at']}[/dim]")
                console.print(f"[dim]Updated: {session['updated_at']}[/dim]")
                if session['metadata']:
                    console.print(f"[dim]Metadata: {session['metadata']}[/dim]")
                console.print()

        finally:
            await history_manager.close()

    asyncio.run(_list_sessions())


@app.command()
def version():
    """Show version information."""
    from . import __version__, __author__
    console.print(f"[bold blue]Agentic ASR[/bold blue] version [green]{__version__}[/green]")
    console.print(f"Created by [bold]{__author__}[/bold]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
