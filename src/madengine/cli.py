"""Click-based CLI entry point for madengine."""

from __future__ import annotations

import json

import click
from omegaconf import OmegaConf

from madengine.config import load_config, merge_overrides
from madengine.pipeline.orchestrator import Orchestrator
from madengine.pipeline.recipes import RecipeRegistry


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """madengine - AI Training Orchestration Engine"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.command()
@click.option("--recipe", "-r", help="Recipe name")
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), help="Config file")
@click.option("--override", "-o", multiple=True, help="Config overrides (key=value)")
@click.option("--profiling", is_flag=True, help="Enable GPU profiling")
@click.option("--dry-run", is_flag=True, help="Preview without execution")
@click.option("--skip-image-build", is_flag=True, help="Skip image building step")
@click.option(
    "--output-format", type=click.Choice(["text", "json"]), default="text", help="Output format"
)
@click.pass_context
def run(
    ctx: click.Context,
    recipe: str | None,
    config_path: str | None,
    override: tuple[str, ...],
    profiling: bool,
    dry_run: bool,
    skip_image_build: bool,
    output_format: str,
) -> None:
    """Run a training job."""
    if not recipe and not config_path:
        raise click.UsageError("Either --recipe or --config is required")

    if recipe:
        cfg = RecipeRegistry().load(recipe)
    else:
        cfg = load_config(config_path)

    cfg = merge_overrides(cfg, override)

    if profiling:
        cfg = OmegaConf.merge(cfg, {"training": {"profiling": {"enabled": True}}})

    cfg.dry_run = dry_run
    cfg.skip_image_build = skip_image_build

    orchestrator = Orchestrator(cfg)
    result = orchestrator.run()

    if output_format == "json":
        click.echo(result.to_json())
    else:
        click.echo(result.to_text())

    ctx.exit(0 if result.success else 1)


@cli.group()
def recipes() -> None:
    """Manage recipes."""


@recipes.command("list")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
def recipes_list(fmt: str) -> None:
    """List available recipes."""
    registry = RecipeRegistry()
    all_recipes = registry.list_all()

    if fmt == "json":
        click.echo(json.dumps([r.to_dict() for r in all_recipes], indent=2))
    else:
        for r in all_recipes:
            click.echo(f"{r.name}: {r.description}")


@recipes.command("show")
@click.argument("name")
def recipes_show(name: str) -> None:
    """Show recipe details."""
    registry = RecipeRegistry()
    recipe = registry.load(name)
    click.echo(OmegaConf.to_yaml(recipe))


@cli.group()
def db() -> None:
    """Database operations."""


@db.command("upload")
@click.option("--metrics-file", "-f", type=click.Path(exists=True), required=True)
@click.option("--backend", type=click.Choice(["mysql", "mongodb"]), default="mongodb")
def db_upload(metrics_file: str, backend: str) -> None:
    """Upload metrics to database."""
    from madengine.database import get_database_backend

    db_backend = get_database_backend({"backend": backend, backend: {}})
    with open(metrics_file) as f:
        metrics = json.load(f)

    if isinstance(metrics, list):
        for record in metrics:
            db_backend.insert(record)
        click.echo(f"Uploaded {len(metrics)} records")
    else:
        db_backend.insert(metrics)
        click.echo("Uploaded 1 record")


@db.command("query")
@click.option("--model", help="Filter by model name")
@click.option("--last", type=int, default=10, help="Number of recent records")
@click.option("--backend", type=click.Choice(["mysql", "mongodb"]), default="mongodb")
def db_query(model: str | None, last: int, backend: str) -> None:
    """Query metrics from database."""
    from madengine.database import get_database_backend

    db_backend = get_database_backend({"backend": backend, backend: {}})
    filters = {}
    if model:
        filters["config.model"] = model

    records = db_backend.query(filters)
    for record in records[:last]:
        record.pop("_id", None)
        click.echo(json.dumps(record, indent=2, default=str))


@cli.group()
def report() -> None:
    """Generate reports."""


@report.command("csv")
@click.option("--query", "-q", help="Filter query (JSON)")
@click.option("--output", "-o", type=click.Path(), required=True)
def report_csv(query: str | None, output: str) -> None:
    """Export metrics to CSV."""
    from madengine.reporting.csv import CSVExporter

    parsed_query = json.loads(query) if query else None
    exporter = CSVExporter()
    exporter.export(query=parsed_query, output_path=output)
    click.echo(f"Exported to {output}")


@report.command("html")
@click.option("--query", "-q", help="Filter query (JSON)")
@click.option("--output", "-o", type=click.Path(), required=True)
def report_html(query: str | None, output: str) -> None:
    """Generate HTML dashboard."""
    from madengine.reporting.html import HTMLExporter

    parsed_query = json.loads(query) if query else None
    exporter = HTMLExporter()
    exporter.export(query=parsed_query, output_path=output)
    click.echo(f"Generated {output}")


@cli.command()
@click.option("--config", "-c", "config_path", type=click.Path(exists=True), required=True)
def validate(config_path: str) -> None:
    """Validate a configuration file."""
    from madengine.config import validate_config

    cfg = load_config(config_path)
    errors = validate_config(cfg)

    if errors:
        click.echo("Validation errors:")
        for err in errors:
            click.echo(f"  - {err}")
        raise SystemExit(1)
    else:
        click.echo("Configuration is valid.")


def main() -> None:
    """Entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
