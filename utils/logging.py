"""
Rich-based logging and visualization utilities for ML training.

Provides:
- Colorful console output with Rich
- Progress bars for training and pipeline stages (including TRL's RichProgressCallback)
- Live metrics display during training
- System information logging (GPU, memory, disk)
- Beautiful tables for configs and results
- File logging with detailed training information

References:
- Rich library: https://rich.readthedocs.io/
- TRL Callbacks: https://huggingface.co/docs/trl/main/en/callbacks
- Transformers Callbacks: https://huggingface.co/docs/transformers/en/main_classes/callback
"""

import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    DownloadColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule
from rich.syntax import Syntax
from rich.tree import Tree
from rich.columns import Columns
from rich.markdown import Markdown
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# Try to import TRL's RichProgressCallback
try:
    from trl.trainer.callbacks import RichProgressCallback as TRLRichProgressCallback
    HAS_TRL_RICH = True
except ImportError:
    HAS_TRL_RICH = False
    TRLRichProgressCallback = None

# Global console instance
console = Console()


def setup_rich_logging(output_dir: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with Rich handler for beautiful terminal output.

    Args:
        output_dir: Directory to save log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create output dir if needed
    os.makedirs(output_dir, exist_ok=True)

    # Configure root logger with Rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            ),
            logging.FileHandler(f"{output_dir}/training.log"),
        ],
        force=True,
    )

    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    return logger


def print_header(text: str, subtitle: Optional[str] = None) -> None:
    """Print a prominent header with optional subtitle."""
    header_text = Text(text, style="bold white")
    if subtitle:
        header_text.append(f"\n{subtitle}", style="dim")

    console.print()
    console.print(Panel(
        header_text,
        style="bold blue",
        border_style="blue",
        padding=(1, 2),
    ))
    console.print()


def print_section(text: str) -> None:
    """Print a section divider."""
    console.print()
    console.print(Rule(text, style="cyan"))
    console.print()


def print_success(message: str) -> None:
    """Print a success message with checkmark."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message with X."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_config_table(config: Any, title: str = "Configuration") -> None:
    """Print a configuration object as a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    # Handle dataclass or dict
    if hasattr(config, "__dataclass_fields__"):
        items = {k: getattr(config, k) for k in config.__dataclass_fields__}
    elif isinstance(config, dict):
        items = config
    else:
        items = vars(config)

    for key, value in items.items():
        # Format value nicely
        if isinstance(value, list):
            value_str = ", ".join(str(v) for v in value)
        elif isinstance(value, bool):
            value_str = "[green]Yes[/green]" if value else "[red]No[/red]"
        elif value is None:
            value_str = "[dim]None[/dim]"
        else:
            value_str = str(value)

        table.add_row(key, value_str)

    console.print(table)
    console.print()


def print_system_info() -> None:
    """Print system information including GPU, memory, and disk."""
    table = Table(title="System Information", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="green")

    # Basic system info
    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("Working Directory", os.getcwd())
    table.add_row("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Try to get GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                table.add_row(f"GPU {i}", f"{gpu_name} ({gpu_mem:.1f} GB)")
            table.add_row("CUDA Version", torch.version.cuda or "N/A")
        else:
            table.add_row("GPU", "[yellow]No CUDA GPU available[/yellow]")
    except ImportError:
        table.add_row("GPU", "[dim]PyTorch not installed[/dim]")

    # Memory info
    try:
        import psutil
        mem = psutil.virtual_memory()
        table.add_row("RAM", f"{mem.total / (1024**3):.1f} GB ({mem.percent}% used)")
        disk = psutil.disk_usage("/")
        table.add_row("Disk", f"{disk.total / (1024**3):.1f} GB ({disk.percent}% used)")
        table.add_row("CPU Cores", str(psutil.cpu_count()))
    except ImportError:
        pass

    console.print(table)
    console.print()


def print_model_info(model: Any) -> None:
    """Print model architecture information."""
    table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # Get model name
    if hasattr(model, "name_or_path"):
        table.add_row("Model Name", model.name_or_path)
    elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        table.add_row("Model Name", model.config._name_or_path)

    # Get architecture
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "architectures"):
            table.add_row("Architecture", ", ".join(config.architectures or ["Unknown"]))
        if hasattr(config, "hidden_size"):
            table.add_row("Hidden Size", str(config.hidden_size))
        if hasattr(config, "num_hidden_layers"):
            table.add_row("Layers", str(config.num_hidden_layers))
        if hasattr(config, "num_attention_heads"):
            table.add_row("Attention Heads", str(config.num_attention_heads))
        if hasattr(config, "vocab_size"):
            table.add_row("Vocab Size", f"{config.vocab_size:,}")

    # Parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Trainable %", f"{100 * trainable_params / total_params:.2f}%")
    except Exception:
        pass

    console.print(table)
    console.print()


def print_dataset_info(dataset: Any, name: str) -> None:
    """Print dataset information."""
    table = Table(title="Dataset Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Dataset", name)
    table.add_row("Samples", f"{len(dataset):,}")

    # Show columns/features
    if hasattr(dataset, "column_names"):
        table.add_row("Columns", ", ".join(dataset.column_names))
    elif hasattr(dataset, "features"):
        table.add_row("Features", ", ".join(dataset.features.keys()))

    # Show sample
    if len(dataset) > 0:
        sample = dataset[0]
        if isinstance(sample, dict):
            sample_keys = list(sample.keys())[:3]
            table.add_row("Sample Keys", ", ".join(sample_keys))

    console.print(table)
    console.print()


def print_training_summary(
    algorithm: str,
    duration: float,
    metrics: Optional[Dict[str, float]] = None,
    status: str = "completed"
) -> None:
    """Print training completion summary."""
    if status == "completed":
        style = "bold green"
        icon = "✓"
    elif status == "failed":
        style = "bold red"
        icon = "✗"
    else:
        style = "bold yellow"
        icon = "⚠"

    table = Table(
        title=f"{icon} {algorithm.upper()} Training {status.title()}",
        show_header=True,
        header_style=style,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Duration
    minutes = int(duration // 60)
    seconds = duration % 60
    if minutes > 0:
        table.add_row("Duration", f"{minutes}m {seconds:.1f}s")
    else:
        table.add_row("Duration", f"{seconds:.1f}s")

    # Add metrics if provided
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))

    console.print(table)
    console.print()


def create_pipeline_progress() -> Progress:
    """Create a progress bar for the overall pipeline."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


class TrainingProgressCallback(TrainerCallback):
    """
    Custom callback for displaying rich training progress.

    Shows:
    - Step progress bar
    - Current metrics (loss, learning rate)
    - GPU memory usage
    - Time estimates
    """

    def __init__(self, algorithm_name: str = "Training"):
        self.algorithm_name = algorithm_name
        self.progress: Optional[Progress] = None
        self.task_id = None
        self.start_time = None
        self.last_log_time = 0
        self.metrics_history: List[Dict[str, float]] = []

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.start_time = time.time()

        console.print()
        console.print(Rule(f"[bold cyan]{self.algorithm_name} Training Started", style="cyan"))
        console.print()

        # Print training args summary
        info_table = Table(show_header=False, box=None)
        info_table.add_column("", style="dim")
        info_table.add_column("", style="cyan")

        info_table.add_row("Max Steps:", str(args.max_steps) if args.max_steps > 0 else "Full epoch")
        info_table.add_row("Batch Size:", str(args.per_device_train_batch_size))
        info_table.add_row("Gradient Accum:", str(args.gradient_accumulation_steps))
        info_table.add_row("Effective Batch:", str(args.per_device_train_batch_size * args.gradient_accumulation_steps))
        info_table.add_row("Learning Rate:", f"{args.learning_rate:.2e}")
        info_table.add_row("Warmup Steps:", str(args.warmup_steps))
        info_table.add_row("Save Strategy:", f"{args.save_strategy} (every {args.save_steps})")

        console.print(info_table)
        console.print()

        # Create progress bar
        total_steps = args.max_steps if args.max_steps > 0 else state.max_steps

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=50, complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]|"),
            TimeElapsedColumn(),
            TextColumn("[dim]<"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )

        self.progress.start()
        self.task_id = self.progress.add_task(
            f"[cyan]{self.algorithm_name}",
            total=total_steps
        )

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when metrics are logged."""
        if logs is None:
            return

        # Store metrics
        self.metrics_history.append(logs.copy())

        # Rate-limit console output (every 2 seconds max)
        current_time = time.time()
        if current_time - self.last_log_time < 2:
            return
        self.last_log_time = current_time

        # Build metrics line
        metrics_parts = []

        if "loss" in logs:
            metrics_parts.append(f"[yellow]loss:[/yellow] {logs['loss']:.4f}")
        if "train_loss" in logs:
            metrics_parts.append(f"[yellow]loss:[/yellow] {logs['train_loss']:.4f}")
        if "learning_rate" in logs:
            metrics_parts.append(f"[blue]lr:[/blue] {logs['learning_rate']:.2e}")
        if "grad_norm" in logs:
            metrics_parts.append(f"[magenta]grad:[/magenta] {logs['grad_norm']:.2f}")
        if "epoch" in logs:
            metrics_parts.append(f"[green]epoch:[/green] {logs['epoch']:.2f}")

        # GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / (1024**3)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                metrics_parts.append(f"[red]GPU:[/red] {mem_used:.1f}/{mem_total:.1f}GB")
        except Exception:
            pass

        if metrics_parts:
            if self.progress:
                # Print above progress bar
                self.progress.console.print(
                    f"    [dim]Step {state.global_step}:[/dim] " + " | ".join(metrics_parts)
                )

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        if self.progress:
            self.progress.console.print(
                f"    [bold green]✓ Checkpoint saved[/bold green] at step {state.global_step}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        if self.progress:
            self.progress.stop()

        duration = time.time() - self.start_time if self.start_time else 0

        console.print()
        console.print(Rule(f"[bold green]{self.algorithm_name} Training Complete", style="green"))

        # Print final stats
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("", style="dim")
        stats_table.add_column("", style="green")

        stats_table.add_row("Total Steps:", str(state.global_step))
        stats_table.add_row("Total Epochs:", f"{state.epoch:.2f}" if state.epoch else "N/A")

        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes > 0:
            stats_table.add_row("Duration:", f"{minutes}m {seconds:.1f}s")
        else:
            stats_table.add_row("Duration:", f"{seconds:.1f}s")

        if state.global_step > 0:
            stats_table.add_row("Steps/sec:", f"{state.global_step / duration:.2f}")

        # Final loss
        if self.metrics_history:
            last_metrics = self.metrics_history[-1]
            if "loss" in last_metrics:
                stats_table.add_row("Final Loss:", f"{last_metrics['loss']:.4f}")
            elif "train_loss" in last_metrics:
                stats_table.add_row("Final Loss:", f"{last_metrics['train_loss']:.4f}")

        console.print(stats_table)
        console.print()


def print_pipeline_results(results: List[tuple]) -> None:
    """Print final pipeline results as a beautiful table."""
    table = Table(
        title="Pipeline Results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Algorithm", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right", style="dim")

    passed = 0
    failed = 0
    total_time = 0

    for algo, status, duration in results:
        total_time += duration

        if status == "PASSED":
            passed += 1
            status_str = "[bold green]PASSED[/bold green]"
        else:
            failed += 1
            status_str = "[bold red]FAILED[/bold red]"

        minutes = int(duration // 60)
        seconds = duration % 60
        if minutes > 0:
            dur_str = f"{minutes}m {seconds:.1f}s"
        else:
            dur_str = f"{seconds:.1f}s"

        table.add_row(algo.upper(), status_str, dur_str)

    # Add totals row
    table.add_row("", "", "")

    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes > 0:
        total_str = f"{minutes}m {seconds:.1f}s"
    else:
        total_str = f"{seconds:.1f}s"

    if failed == 0:
        summary = f"[bold green]{passed} passed[/bold green]"
    else:
        summary = f"[bold green]{passed} passed[/bold green], [bold red]{failed} failed[/bold red]"

    table.add_row("[bold]TOTAL[/bold]", summary, f"[bold]{total_str}[/bold]")

    console.print()
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


def print_step_info(step: int, total_steps: int, metrics: Dict[str, Any]) -> None:
    """Print a single step's information with metrics."""
    progress_pct = (step / total_steps * 100) if total_steps > 0 else 0

    # Build metrics string
    metrics_parts = []
    if "loss" in metrics:
        metrics_parts.append(f"[yellow]loss={metrics['loss']:.4f}[/yellow]")
    if "learning_rate" in metrics:
        metrics_parts.append(f"[blue]lr={metrics['learning_rate']:.2e}[/blue]")
    if "grad_norm" in metrics:
        metrics_parts.append(f"[magenta]grad={metrics['grad_norm']:.3f}[/magenta]")

    metrics_str = " | ".join(metrics_parts) if metrics_parts else ""

    console.print(
        f"  [dim]Step[/dim] [cyan]{step:,}[/cyan]/[dim]{total_steps:,}[/dim] "
        f"([green]{progress_pct:.1f}%[/green]) {metrics_str}"
    )


def print_gpu_memory_usage() -> None:
    """Print current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                console.print(
                    f"  [red]GPU {i}:[/red] "
                    f"[yellow]{allocated:.2f}GB[/yellow] allocated / "
                    f"[orange3]{reserved:.2f}GB[/orange3] reserved / "
                    f"[dim]{total:.1f}GB total[/dim]"
                )
    except Exception as e:
        console.print(f"  [dim]GPU memory info unavailable: {e}[/dim]")


def print_checkpoint_saved(path: str, step: int) -> None:
    """Print checkpoint saved notification."""
    console.print(
        f"  [bold green]✓[/bold green] [green]Checkpoint saved[/green] "
        f"at step [cyan]{step:,}[/cyan] → [dim]{path}[/dim]"
    )


def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """Print evaluation results in a formatted table."""
    table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


class DetailedLoggingCallback(TrainerCallback):
    """
    Callback that logs detailed training information to files.

    Creates:
    - training_metrics.jsonl: JSON lines file with all metrics
    - training_events.log: Human-readable event log
    """

    def __init__(self, output_dir: str, algorithm_name: str = "Training"):
        self.output_dir = Path(output_dir)
        self.algorithm_name = algorithm_name
        self.metrics_file = None
        self.events_file = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        import json
        self.start_time = time.time()

        # Open log files
        self.metrics_file = open(self.output_dir / "training_metrics.jsonl", "w")
        self.events_file = open(self.output_dir / "training_events.log", "w")

        # Log start event
        event = {
            "event": "train_start",
            "timestamp": datetime.now().isoformat(),
            "algorithm": self.algorithm_name,
            "max_steps": args.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
        }
        self.events_file.write(f"{datetime.now().isoformat()} | TRAIN_START | {json.dumps(event)}\n")
        self.events_file.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        import json
        if logs and self.metrics_file:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                **logs,
            }
            self.metrics_file.write(json.dumps(log_entry) + "\n")
            self.metrics_file.flush()

    def on_save(self, args, state, control, **kwargs):
        import json
        if self.events_file:
            event = {
                "event": "checkpoint_saved",
                "step": state.global_step,
                "timestamp": datetime.now().isoformat(),
            }
            self.events_file.write(f"{datetime.now().isoformat()} | CHECKPOINT | Step {state.global_step}\n")
            self.events_file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        import json
        duration = time.time() - self.start_time if self.start_time else 0

        if self.events_file:
            event = {
                "event": "train_end",
                "timestamp": datetime.now().isoformat(),
                "total_steps": state.global_step,
                "duration_seconds": duration,
            }
            self.events_file.write(f"{datetime.now().isoformat()} | TRAIN_END | Duration: {format_duration(duration)}, Steps: {state.global_step}\n")
            self.events_file.close()

        if self.metrics_file:
            self.metrics_file.close()


class VerboseLoggingCallback(TrainerCallback):
    """
    Callback that prints verbose information about every training event.

    Use this when you want to see EVERYTHING that's happening.
    """

    def __init__(self, algorithm_name: str = "Training"):
        self.algorithm_name = algorithm_name
        self.step_times: List[float] = []
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        console.print()
        console.print(Panel(
            f"[bold cyan]{self.algorithm_name}[/bold cyan] Training Initialized",
            subtitle=f"[dim]Max steps: {args.max_steps if args.max_steps > 0 else 'auto'}[/dim]",
            style="cyan"
        ))

        # Print all training arguments
        print_section("Training Arguments")
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith("_")}
        print_config_table(args_dict, "TrainingArguments")

        self.last_step_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self.last_step_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_step_time:
            step_duration = time.time() - self.last_step_time
            self.step_times.append(step_duration)

            # Print step timing every 10 steps
            if state.global_step % 10 == 0 and self.step_times:
                avg_time = sum(self.step_times[-10:]) / min(10, len(self.step_times))
                remaining_steps = (args.max_steps - state.global_step) if args.max_steps > 0 else 0
                eta = remaining_steps * avg_time

                console.print(
                    f"  [dim]Step {state.global_step}[/dim] | "
                    f"[cyan]{step_duration:.3f}s/step[/cyan] | "
                    f"[yellow]avg: {avg_time:.3f}s[/yellow] | "
                    f"[green]ETA: {format_duration(eta)}[/green]"
                )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print_step_info(state.global_step, args.max_steps if args.max_steps > 0 else state.max_steps, logs)
            print_gpu_memory_usage()

    def on_save(self, args, state, control, **kwargs):
        print_checkpoint_saved(args.output_dir, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print_section("Evaluation")
            print_evaluation_results(metrics)

    def on_train_end(self, args, state, control, **kwargs):
        console.print()
        console.print(Panel(
            f"[bold green]{self.algorithm_name}[/bold green] Training Complete!",
            subtitle=f"[dim]Total steps: {state.global_step}[/dim]",
            style="green"
        ))

        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            console.print(f"  [dim]Average step time:[/dim] [cyan]{avg_time:.3f}s[/cyan]")
            console.print(f"  [dim]Total steps:[/dim] [cyan]{len(self.step_times)}[/cyan]")


def get_training_callbacks(
    output_dir: str,
    algorithm_name: str = "Training",
    verbose: bool = True,
    use_trl_rich: bool = True,
) -> List[TrainerCallback]:
    """
    Get a list of training callbacks for comprehensive logging.

    Args:
        output_dir: Directory to save logs
        algorithm_name: Name of the algorithm for display
        verbose: Whether to use verbose console logging
        use_trl_rich: Whether to use TRL's RichProgressCallback if available

    Returns:
        List of TrainerCallback instances
    """
    callbacks = []

    # Always add detailed file logging
    callbacks.append(DetailedLoggingCallback(output_dir, algorithm_name))

    # Add our custom progress callback
    callbacks.append(TrainingProgressCallback(algorithm_name))

    # Add verbose logging if requested
    if verbose:
        callbacks.append(VerboseLoggingCallback(algorithm_name))

    # Add TRL's RichProgressCallback if available and requested
    if use_trl_rich and HAS_TRL_RICH and TRLRichProgressCallback is not None:
        try:
            callbacks.append(TRLRichProgressCallback())
        except Exception:
            pass  # Silently skip if it fails

    return callbacks


def print_script_header(script_name: str, description: str = "") -> None:
    """Print a header for a script execution."""
    console.print()
    console.print(Panel(
        Text.assemble(
            (script_name, "bold white"),
            "\n",
            (description, "dim") if description else ("", ""),
        ),
        title="[bold blue]Script Started[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))
    print_system_info()


def print_script_footer(
    script_name: str,
    success: bool,
    duration: float,
    results: Optional[List[tuple]] = None
) -> None:
    """Print a footer for script completion."""
    if success:
        style = "green"
        icon = "✓"
        status = "Completed Successfully"
    else:
        style = "red"
        icon = "✗"
        status = "Failed"

    console.print()
    console.print(Panel(
        Text.assemble(
            (f"{icon} {status}", f"bold {style}"),
            "\n",
            (f"Duration: {format_duration(duration)}", "dim"),
        ),
        title=f"[bold {style}]{script_name}[/bold {style}]",
        border_style=style,
        padding=(1, 2),
    ))

    if results:
        print_pipeline_results(results)
