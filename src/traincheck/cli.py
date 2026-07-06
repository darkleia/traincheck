"""Console script for traincheck."""

import json
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from traincheck.core import Severity
from traincheck.validator import Validator

app = typer.Typer(
    name="traincheck",
    help="A configuration linter for distributed GPU training.",
    no_args_is_help=True,
)
console = Console()

ICONS = {Severity.ERROR: "❌", Severity.WARN: "⚠️", Severity.INFO: "ℹ️"}
COLORS = {Severity.ERROR: "red", Severity.WARN: "yellow", Severity.INFO: "blue"}


@app.command()
def check(
    config_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Path to a traincheck config YAML file."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Print results as JSON instead of a table."
    ),
) -> None:
    """Validate a training configuration against known failure patterns."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    result = Validator().validate(config)

    if json_output:
        _print_json(result)
    else:
        _print_table(config_path, result)

    if not result.passed:
        raise typer.Exit(code=1)


def _print_table(config_path: Path, result) -> None:
    if not result.violations:
        console.print(f"\n✅ All checks passed. No issues found in {config_path}.\n")
        return

    table = Table(show_header=False, box=None, padding=(0, 1))
    for violation in result.violations:
        rule = violation.rule
        icon, color = ICONS[rule.severity], COLORS[rule.severity]
        table.add_row(
            icon, f"[{color}][{rule.severity.value.upper()}] {rule.id}[/{color}]: {rule.message}"
        )
        if rule.fix_suggestion:
            table.add_row("", f"[dim]Fix: {rule.fix_suggestion}[/dim]")
    console.print(table)

    errors, warnings = len(result.errors), len(result.warnings)
    infos = len(result.violations) - errors - warnings
    color = "red" if errors else "yellow"
    console.print(f"\n[{color}]{errors} error(s), {warnings} warning(s), {infos} info(s)[/{color}]\n")


def _print_json(result) -> None:
    output = {
        "passed": result.passed,
        "violations": [
            {
                "id": v.rule.id,
                "severity": v.rule.severity.value,
                "message": v.rule.message,
                "fix_suggestion": v.rule.fix_suggestion,
            }
            for v in result.violations
        ],
    }
    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
