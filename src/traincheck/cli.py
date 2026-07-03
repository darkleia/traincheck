"""Console script for traincheck."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from traincheck.validator import Validator
from traincheck.core import Severity

app = typer.Typer(
    name="traincheck",
    help="A configuration linter for distributed GPU training.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

SEVERITY_COLORS = {
    Severity.ERROR: "red",
    Severity.WARN: "yellow",
    Severity.INFO: "blue",
}

SEVERITY_ICONS = {
    Severity.ERROR: "❌",
    Severity.WARN: "⚠️",
    Severity.INFO: "ℹ️",
}


@app.command()
def check(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to traincheck config YAML file.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON instead of a table.",
    ),
) -> None:
    """Validate a training configuration against known failure patterns."""
    config = _load_config(config_path)
    validator = Validator()
    result = validator.validate(config)

    if json_output:
        _print_json(result)
    else:
        _print_table(config_path, result)

    if not result.passed:
        raise typer.Exit(code=1)


def _load_config(path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _print_table(config_path: Path, result) -> None:
    """Print validation results as a rich table."""
    total = len(result.violations)

    if total == 0:
        console.print(
            f"\n✅ [bold green]All checks passed.[/bold green] "
            f"No issues found in [bold]{config_path}[/bold].\n"
        )
        return

    console.print(
        f"\n🔍 Running {len(result.violations)} checks on [bold]{config_path}[/bold]..."
    )

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("icon", width=2)
    table.add_column("output", ratio=1)

    for violation in result.violations:
        rule = violation.rule
        icon = SEVERITY_ICONS.get(rule.severity, "•")
        color = SEVERITY_COLORS.get(rule.severity, "white")

        table.add_row(
            icon,
            f"[bold {color}][{rule.severity.value.upper()}] {rule.id}[/bold {color}]: "
            f"{rule.message}",
        )
        if rule.fix_suggestion:
            table.add_row(
                "",
                f"[dim]💡 Fix: {rule.fix_suggestion}[/dim]",
            )

    console.print(table)

    summary_color = "red" if result.errors else "yellow"
    infos = len([v for v in result.violations if v.rule.severity == Severity.INFO])
    console.print(
        f"\n{'❌' if result.errors else '✅'} "
        f"[bold {summary_color}]{len(result.errors)} error(s), "
        f"{len(result.warnings)} warning(s), "
        f"{infos} info(s)[/bold {summary_color}]"
    )

    if not result.passed:
        console.print(
            "\n[bold red]❌ Critical errors found. Fix them before launching.[/bold red]\n"
        )
    else:
        console.print("\n[bold green]✅ Ready to launch.[/bold green]\n")


def _print_json(result) -> None:
    """Print validation results as JSON."""
    import json

    output = {
        "passed": result.passed,
        "errors": len(result.errors),
        "warnings": len(result.warnings),
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