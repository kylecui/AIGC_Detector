import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aigc_detector.models.registry import get_registry, get_models_by_purpose


def download_model(model_name: str, model_info, models_dir: Path, console: Console) -> None:
    """Download a single model from HuggingFace."""
    model_path = models_dir / model_name

    # Skip if local_path exists and is non-empty
    if model_info.local_path:
        local_check = Path(model_info.local_path)
        if local_check.exists() and any(local_check.iterdir()):
            console.print(f"[yellow]⊘[/yellow] {model_name}: Using existing local path {model_info.local_path}")
            return

    # Skip if model directory exists and is non-empty
    if model_path.exists() and any(model_path.iterdir()):
        console.print(f"[yellow]⊘[/yellow] {model_name}: Already downloaded to {model_path}")
        return

    console.print(f"[cyan]↓[/cyan] Downloading {model_name} ({model_info.hf_id})...")

    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_info.hf_id,
            local_dir=str(model_path),
            repo_type="model",
            resume_download=True,
        )
        console.print(f"[green]✓[/green] {model_name}: Downloaded to {model_path}")
    except Exception as e:
        console.print(f"[red]✗[/red] {model_name}: Failed to download - {e}")


def main():
    parser = argparse.ArgumentParser(description="Download models for AIGC Detector")
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model name to download",
    )
    parser.add_argument(
        "--purpose",
        type=str,
        help="Download all models with a specific purpose (e.g., generation, statistical, encoder)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models (default: models)",
    )

    args = parser.parse_args()

    console = Console()
    registry = get_registry()
    models_to_download = {}

    if args.model:
        if args.model not in registry:
            console.print(f"[red]Error:[/red] Model '{args.model}' not found in registry")
            return 1
        models_to_download[args.model] = registry[args.model]

    elif args.purpose:
        models_by_purpose = get_models_by_purpose(args.purpose)
        if not models_by_purpose:
            console.print(f"[red]Error:[/red] No models found with purpose '{args.purpose}'")
            return 1
        for model in models_by_purpose:
            models_to_download[model.name] = model

    elif args.all:
        models_to_download = registry

    else:
        parser.print_help()
        return 1

    console.print(f"[bold]Downloading {len(models_to_download)} model(s)[/bold]")

    for model_name, model_info in models_to_download.items():
        download_model(model_name, model_info, args.models_dir, console)

    console.print("[bold green]Done![/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
