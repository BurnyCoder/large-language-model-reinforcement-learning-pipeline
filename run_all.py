"""
Run all training pipelines with Rich visualization.

Usage:
    python run_all.py              # Run both test and production pipelines
    python run_all.py --test       # Run only test pipeline (fast)
    python run_all.py --prod       # Run only production pipeline

Features:
- Rich console output with colors and progress bars
- System information display
- Detailed timing and status for each script
- Beautiful result tables
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.rule import Rule
from rich.live import Live

# Import DualConsole for file logging support
from utils.logging import DualConsole

console = DualConsole()

# Log directory for this run (set in main())
_log_dir: Optional[Path] = None


def print_header(mode: str, scripts: List[str]) -> None:
    """Print a beautiful header for the run."""
    console.print()
    console.print(Panel(
        Text.assemble(
            ("LLM RL Training Pipelines", "bold white"),
            "\n\n",
            (f"Mode: {mode}", "cyan"),
            "\n",
            (f"Scripts: {', '.join(scripts)}", "dim"),
        ),
        title="[bold blue]Run All[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def print_system_info() -> None:
    """Print system information."""
    import platform
    from datetime import datetime

    table = Table(title="System Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("Working Directory", str(Path.cwd()))
    table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                table.add_row(f"GPU {i}", f"{gpu_name} ({gpu_mem:.1f} GB)")
        else:
            table.add_row("GPU", "[yellow]No CUDA GPU available[/yellow]")
    except ImportError:
        table.add_row("GPU", "[dim]PyTorch not installed[/dim]")

    console.print(table)
    console.print()


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def run_script(script_name: str, script_dir: Path) -> Tuple[str, int, float]:
    """Run a script and return (name, return_code, duration).

    Captures all subprocess output to both console (terminal + master log)
    and an individual script log file.
    """
    console.print()
    console.print(Rule(f"[bold cyan]Running {script_name}[/bold cyan]", style="cyan"))
    console.print()

    console.print(f"[dim]Command:[/dim] python {script_dir / script_name}")
    console.print(f"[dim]Working directory:[/dim] {script_dir}")
    console.print()

    start = time.time()

    # Run with real-time output
    process = subprocess.Popen(
        [sys.executable, script_dir / script_name],
        cwd=script_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream output to console and individual script log file
    script_log_file = None
    if _log_dir:
        script_log_path = _log_dir / f"{script_name.replace('.py', '')}_output.log"
        script_log_file = open(script_log_path, 'w', encoding='utf-8')

    try:
        if process.stdout:
            for line in process.stdout:
                line_stripped = line.rstrip()
                console.print(line_stripped)  # Goes to terminal + master log
                if script_log_file:
                    script_log_file.write(line)
                    script_log_file.flush()
    finally:
        if script_log_file:
            script_log_file.close()

    process.wait()
    duration = time.time() - start

    console.print()
    if process.returncode == 0:
        console.print(f"[bold green]✓[/bold green] {script_name} completed in {format_duration(duration)}")
    else:
        console.print(f"[bold red]✗[/bold red] {script_name} failed with code {process.returncode}")

    return script_name, process.returncode, duration


def print_results(results: List[Tuple[str, str, float]], total_duration: float, mode: str) -> None:
    """Print final results table."""
    console.print()
    console.print(Rule("[bold magenta]Pipeline Results[/bold magenta]", style="magenta"))
    console.print()

    table = Table(title=f"{mode} Pipeline Results", show_header=True, header_style="bold magenta")
    table.add_column("Script", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right", style="dim")

    passed = 0
    failed = 0

    for script, status, duration in results:
        if status == "PASSED":
            passed += 1
            status_str = "[bold green]PASSED[/bold green]"
        else:
            failed += 1
            status_str = "[bold red]FAILED[/bold red]"

        table.add_row(script, status_str, format_duration(duration))

    # Add totals row
    table.add_row("", "", "")

    if failed == 0:
        summary = f"[bold green]{passed} passed[/bold green]"
    else:
        summary = f"[bold green]{passed} passed[/bold green], [bold red]{failed} failed[/bold red]"

    table.add_row("[bold]TOTAL[/bold]", summary, f"[bold]{format_duration(total_duration)}[/bold]")

    console.print(table)
    console.print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM RL training pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py          # Run both test and production pipelines
  python run_all.py --test   # Run only test pipeline (fast)
  python run_all.py --prod   # Run only production pipeline
        """
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run only test pipeline (tiny_gpt2)",
    )
    parser.add_argument(
        "--prod",
        "-p",
        action="store_true",
        help="Run only production pipeline (qwen2.5_0.5)",
    )
    return parser.parse_args()


def main():
    global _log_dir

    args = parse_args()
    script_dir = Path(__file__).parent

    # Create log directory for run_all output
    _log_dir = script_dir / "logs" / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _log_dir.mkdir(parents=True, exist_ok=True)

    # Configure console to write all output to master log file
    master_log = _log_dir / "master_output.log"
    console.set_log_file(str(master_log))

    # Determine which pipelines to run
    if args.test and args.prod:
        scripts = ["test_models.py", "qwen2.5_0.5.py"]
        mode = "ALL"
    elif args.test:
        scripts = ["test_models.py"]
        mode = "TEST"
    elif args.prod:
        scripts = ["qwen2.5_0.5.py"]
        mode = "PRODUCTION"
    else:
        scripts = ["test_models.py", "qwen2.5_0.5.py"]
        mode = "ALL"

    # Print header
    print_header(mode, scripts)

    # Print system info
    print_system_info()

    # Run scripts
    start_time = time.time()
    results: List[Tuple[str, str, float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console._terminal,  # Use terminal console for progress bar
    ) as progress:
        overall_task = progress.add_task(f"[cyan]Running {mode} pipelines...", total=len(scripts))

        for script in scripts:
            progress.update(overall_task, description=f"[cyan]Running {script}...")

            name, returncode, duration = run_script(script, script_dir)
            status = "PASSED" if returncode == 0 else "FAILED"
            results.append((name, status, duration))

            progress.advance(overall_task)

            if returncode != 0:
                console.print(f"[bold red]Error:[/bold red] {script} failed with return code {returncode}")

    total_duration = time.time() - start_time

    # Print results
    print_results(results, total_duration, mode)

    # Final summary
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")

    console.print()
    if failed == 0:
        console.print(Panel(
            Text.assemble(
                ("All pipelines completed successfully!", "bold green"),
                "\n",
                (f"Total time: {format_duration(total_duration)}", "dim"),
            ),
            title="[bold green]Success[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            Text.assemble(
                (f"{failed} pipeline(s) failed", "bold red"),
                "\n",
                (f"{passed} passed, {failed} failed", "dim"),
            ),
            title="[bold red]Failure[/bold red]",
            border_style="red",
            padding=(1, 2),
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
