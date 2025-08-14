"""CLI commands for model management."""

import json
from typing import Any, Dict, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table

from medvllm.engine.model_runner import registry

console = Console()


@click.group()
def model() -> None:
    """Manage models in the registry."""
    pass


# Import ModelType from the registry
from medvllm.engine.model_runner.registry import ModelType


@model.command()
@click.argument("name")
@click.argument("path")
@click.option(
    "--type",
    "model_type_str",
    type=click.Choice([t.name.lower() for t in ModelType]),
    default="generic",
    help="Type of the model",
)
@click.option("--description", help="Description of the model")
@click.option("--tag", "tags", multiple=True, help="Tags for the model")
@click.option("--param", "params", multiple=True, help="Additional parameters (key=value)")
def register(
    name: str,
    path: str,
    model_type_str: str,
    description: str,
    tags: Tuple[str, ...],
    params: Tuple[str, ...],
) -> None:
    """Register a new model in the registry."""
    try:
        # Parse parameters
        params_dict: Dict[str, str] = {}
        for p in params:
            if "=" not in p:
                raise click.BadParameter(f"Invalid parameter format: {p}. Expected key=value")
            key, value = p.split("=", 1)
            params_dict[key] = value

        # Get the ModelType enum value
        model_type_enum = ModelType[model_type_str.upper()]

        # Register the model with the registry
        # Get tags as list of strings
        tag_list = list(tags) if tags else None

        # Extract and validate loader if it exists in params_dict
        loader = None
        if params_dict and "loader" in params_dict:
            loader_str = params_dict.pop("loader")
            if isinstance(loader_str, str):
                try:
                    # Try to import the loader class dynamically
                    from importlib import import_module

                    module_path, class_name = loader_str.rsplit(".", 1)
                    module = import_module(module_path)
                    loader = getattr(module, class_name)
                    if not (isinstance(loader, type) and hasattr(loader, "load_model")):
                        console.print(
                            f"[yellow]Warning: {loader_str} is not a valid loader class, ignoring[/]"
                        )
                        loader = None
                except (ImportError, AttributeError, ValueError) as e:
                    console.print(
                        f"[yellow]Warning: Could not load loader {loader_str}: {e}, ignoring[/]"
                    )
                    loader = None

        # Include path in the parameters dictionary
        model_params = {"pretrained_model_name_or_path": path}
        if params_dict:
            model_params.update(params_dict)

        # Register the model with explicit parameters
        registry.register(
            name=name,
            model_type=model_type_enum,
            model_class=None,  # Will be loaded from path
            config_class=None,  # Will be inferred
            description=description or "",
            tags=tag_list,
            loader=loader,
            parameters=model_params,
        )
        console.print(f"[green]✓[/] Registered model: {name}")
    except Exception as e:
        console.print(f"[red]✗ Failed to register model: {str(e)}[/]")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.option("--force", is_flag=True, help="Force unregister even if in use")
def unregister(name: str, force: bool) -> None:
    """Remove a model from the registry."""
    try:
        registry.unregister(name)
        console.print(f"[green]✓[/] Unregistered model: {name}")
    except KeyError:
        console.print(f"[yellow]![/] Model not found: {name}")
    except Exception as e:
        if force:
            registry._models.pop(name, None)  # Force remove
            console.print(f"[yellow]![/] Force unregistered model: {name}")
        else:
            console.print(f"[red]✗ Failed to unregister model: {str(e)}[/]")
            console.print("Use --force to force unregister")
            raise click.Abort()


@model.command("list")
@click.option("--type", "model_type_str", help="Filter by model type")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def list_models(model_type_str: Optional[str], output_json: bool) -> None:
    """List all registered models.

    Args:
        model_type_str: Optional model type to filter by
        output_json: If True, output as JSON instead of a table
    """
    try:
        model_type_enum = ModelType[model_type_str.upper()] if model_type_str else None
        models = registry.list_models(model_type_enum)

        if output_json:
            # Ensure all values are JSON-serializable
            serializable_models = []
            for model in models:
                serializable_model = {}
                for key, value in model.items():
                    if hasattr(value, "name") and hasattr(value, "value"):  # Handle enums
                        serializable_model[key] = value.name.lower()
                    else:
                        serializable_model[key] = value
                serializable_models.append(serializable_model)
            click.echo(json.dumps(serializable_models, indent=2))
            return

        table = Table(title="Registered Models")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description")
        table.add_column("Tags")

        for model_info in models:
            table.add_row(
                model_info.get("name", ""),
                model_info.get("model_type", ""),
                model_info.get("description", ""),
                ", ".join(model_info.get("tags", [])),
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Failed to list models: {str(e)}[/]")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def info(name: str, output: Optional[str]) -> None:
    """Show detailed information about a model."""
    try:
        meta = registry.get_metadata(name)
        info = {
            "name": meta.name,
            "type": meta.model_type.name.lower(),
            "description": meta.description,
            "tags": meta.tags,
            "parameters": meta.parameters,
            "model_class": meta.model_class.__name__ if meta.model_class else None,
            "config_class": meta.config_class.__name__ if meta.config_class else None,
        }

        if output:
            with open(output, "w") as f:
                json.dump(info, f, indent=2)
            console.print(f"[green]✓[/] Model info saved to {output}")
        else:
            console.print_json(data=info)
    except KeyError:
        console.print(f"[red]✗ Model not found: {name}[/]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]✗ Failed to get model info: {str(e)}[/]")
        raise click.Abort()


@model.command()
def clear_cache() -> None:
    """Clear the model cache."""
    try:
        registry.clear_cache()
        console.print("[green]✓[/] Model cache cleared")
    except Exception as e:
        console.print(f"[red]✗ Failed to clear cache: {str(e)}[/]")
        raise click.Abort()


def register_commands(cli: Any) -> None:
    """Register model commands with the main CLI."""
    cli.add_command(model)
