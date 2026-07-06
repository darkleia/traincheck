"""Console script for traincheck."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from traincheck.core import Result, Severity
from traincheck.resolve import UnsupportedStackError, resolve
from traincheck.validator import Validator
from traincheck.verification import VerificationItem, collect_needs_verification

app = typer.Typer(
    name="traincheck",
    help="A configuration linter for distributed GPU training.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

ICONS = {Severity.ERROR: "❌", Severity.WARN: "⚠️", Severity.INFO: "ℹ️"}
COLORS = {Severity.ERROR: "red", Severity.WARN: "yellow", Severity.INFO: "blue"}


@app.callback()
def main() -> None:
    """A configuration linter for distributed GPU training."""


@app.command()
def check(
    config_path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to a training job config: native YAML, an sbatch script, or another supported launcher file.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Print results as JSON instead of a table."
    ),
) -> None:
    """Validate a training job against known failure patterns."""
    try:
        spec = resolve(str(config_path))
    except UnsupportedStackError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    result = Validator().validate_spec(spec)
    verification_items = collect_needs_verification(spec, result)

    if json_output:
        _print_json(result, verification_items)
    else:
        _print_table(config_path, result, verification_items)

    if not result.passed:
        raise typer.Exit(code=1)


def _print_table(config_path: Path, result: Result, verification_items: list) -> None:
    console.print(f"\n🔍 Checked [bold]{config_path}[/bold]")

    console.print("\n[bold]Violations[/bold]")
    if not result.violations:
        console.print("  none")
    else:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("icon", width=2)
        table.add_column("output", ratio=1)
        for violation in result.violations:
            rule = violation.rule
            icon, color = ICONS[rule.severity], COLORS[rule.severity]
            table.add_row(
                icon, f"[{color}][{rule.severity.value.upper()}] {rule.id}[/{color}]: {rule.message}"
            )
            if rule.fix_suggestion:
                table.add_row("", f"[dim]Fix: {rule.fix_suggestion}[/dim]")
        console.print(table)

    console.print("\n[bold]Needs verification[/bold]")
    if not verification_items:
        console.print("  none")
    else:
        for item in verification_items:
            label = f"[dim]({item.rule_id})[/dim] " if item.rule_id else ""
            console.print(f"  ⚠️  {label}{item.display}")

    errors, warnings = len(result.errors), len(result.warnings)
    infos = len(result.violations) - errors - warnings
    color = "red" if errors else "yellow"
    console.print("\n[bold]Summary[/bold]")
    console.print(
        f"  [{color}]{errors} error(s), {warnings} warning(s), {infos} info(s)[/{color}], "
        f"{len(verification_items)} to verify\n"
    )


def _print_json(result: Result, verification_items: list[VerificationItem]) -> None:
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
        "needs_verification": [
            {
                "rule_id": item.rule_id,
                "field": item.field_name,
                "reason": item.reason,
                "check_command": item.check_command,
            }
            for item in verification_items
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
