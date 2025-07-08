"""CLI commands for model management."""

import json
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from medvllm.engine.model_runner import registry

console = Console()


@click.group()
def model():
    """Manage models in the registry."""
    pass


@model.command()
@click.argument("name")
@click.argument("path")
@click.option("--type", "model_type", type=click.Choice([t.name.lower() for t in registry.ModelType]),
              default="generic", help="Type of the model")
@click.option("--description", help="Description of the model")
@click.option("--tag", "tags", multiple=True, help="Tags for the model")
@click.option("--param", "params", multiple=True, help="Additional parameters (key=value)")
def register(name: str, path: str, model_type: str, description: str, tags: tuple, params: tuple):
    """Register a new model in the registry."""
    try:
        # Parse parameters
        params_dict = {}
        for p in params:
            if "=" not in p:
                raise click.BadParameter(f"Invalid parameter format: {p}. Expected key=value")
            key, value = p.split("=", 1)
            params_dict[key] = value
        
        # Register the model
        registry.register(
            name=name,
            model_type=registry.ModelType[model_type.upper()],
            model_path=path,
            description=description or "",
            tags=list(tags),
            **params_dict
        )
        console.print(f"[green]✓[/] Registered model: {name}")
    except Exception as e:
        console.print(f"[red]✗ Failed to register model: {str(e)}[/]")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.option("--force", is_flag=True, help="Force unregister even if in use")
def unregister(name: str, force: bool):
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
@click.option("--type", "model_type", type=click.Choice([t.name.lower() for t in registry.ModelType]),
              help="Filter by model type")
@click.option("--json", is_flag=True, help="Output as JSON")
def list_models(model_type: Optional[str], json: bool):
    """List all registered models."""
    try:
        model_type_enum = registry.ModelType[model_type.upper()] if model_type else None
        models = registry.list_models(model_type_enum)
        
        if json:
            # Output as JSON
            output = []
            for name, meta in models.items():
                output.append({
                    "name": name,
                    "type": meta.model_type.name.lower(),
                    "description": meta.description,
                    "tags": meta.tags,
                    "parameters": meta.parameters
                })
            console.print_json(data=output)
        else:
            # Output as formatted table
            table = Table(title="Registered Models")
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Description")
            table.add_column("Tags")
            
            for name, meta in models.items():
                table.add_row(
                    name,
                    meta.model_type.name.lower(),
                    meta.description,
                    ", ".join(meta.tags) if meta.tags else ""
                )
            
            console.print(table)
    except Exception as e:
        console.print(f"[red]✗ Failed to list models: {str(e)}[/]")
        raise click.Abort()


@model.command()
@click.argument("name")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def info(name: str, output: Optional[str]):
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
            "config_class": meta.config_class.__name__ if meta.config_class else None
        }
        
        if output:
            with open(output, 'w') as f:
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
def clear_cache():
    """Clear the model cache."""
    try:
        registry.clear_cache()
        console.print("[green]✓[/] Model cache cleared")
    except Exception as e:
        console.print(f"[red]✗ Failed to clear cache: {str(e)}[/]")
        raise click.Abort()


def register_commands(cli):
    """Register model commands with the main CLI."""
    cli.add_command(model)
