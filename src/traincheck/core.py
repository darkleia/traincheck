"""Core rule engine for traincheck."""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from traincheck.ir import Field
from traincheck.utils import dependency_constraint, parse_pinned_version


class Severity(Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


# A condition/detail expression runs with no real __builtins__ (no file/network/
# import access), but a handful of plain type-coercion functions are common and
# safe enough to allow explicitly - e.g. "str(gpu_type).startswith('H100')".
# Without this, calling any of them raises NameError, and `evaluate`'s except
# swallows that into a silent "condition is False" - a rule using one of these
# would look registered and correct but could never actually fire.
# parse_pinned_version/dependency_constraint are here for the same reason: a
# condition that reads dependency_constraints (a raw {package: constraint}
# dict, keys spelled however the requirements.txt/lockfile wrote them) needs
# a normalized lookup and a way to turn e.g. "==1.13.0" into (1, 13, 0), and
# these are the two safe, narrowly-scoped functions that do it.
_SAFE_BUILTINS: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sorted": sorted,
    "parse_pinned_version": parse_pinned_version,
    "dependency_constraint": dependency_constraint,
}


@dataclass
class Rule:
    id: str
    severity: Severity
    condition: str
    message: str
    fix_suggestion: Optional[str] = None
    # An optional Python expression (evaluated the same way `condition` is)
    # producing a short, value-specific supplement to the static `message` -
    # e.g. the actual minAvailable/replica counts that tripped a
    # gang-scheduling rule, rather than just the generic rule description.
    detail: Optional[str] = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the rule's condition against the given context.
        Returns True if the rule is violated (i.e., the condition is true).
        """
        try:
            return bool(eval(self.condition, {"__builtins__": _SAFE_BUILTINS}, context))
        except Exception:
            return False

    def render_detail(self, context: dict[str, Any]) -> Optional[str]:
        if self.detail is None:
            return None
        try:
            return str(eval(self.detail, {"__builtins__": _SAFE_BUILTINS}, context))
        except Exception:
            return None


@dataclass
class Violation:
    rule: Rule
    context: dict[str, Any]
    detail: Optional[str] = None


@dataclass
class NeedsVerification:
    rule: Rule
    field_name: str
    reason: str


@dataclass
class Result:
    violations: list[Violation] = field(default_factory=list)
    needs_verification: list[NeedsVerification] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.WARN]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


def _condition_names(condition: str) -> set[str]:
    """Names a rule condition reads, e.g. {"nodes", "gpu_type"} for
    "nodes > 32 and gpu_type == 'A100'".
    """
    tree = ast.parse(condition, mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


class RuleEngine:
    def __init__(self):
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)

    def check(self, context: dict[str, Any]) -> Result:
        """Run every registered rule against context.

        Context values may be plain values or `Field`s. A rule whose
        condition reads a `Field` with status "unknown" is not evaluated -
        we don't know the value, so we can't safely conclude pass or fail -
        and instead is reported via `Result.needs_verification`.
        """
        violations = []
        needs_verification = []
        flat = {name: value.value if isinstance(value, Field) else value for name, value in context.items()}

        for rule in self._rules:
            try:
                names = _condition_names(rule.condition)
            except SyntaxError:
                names = set()

            unresolved = next(
                (
                    (name, context[name])
                    for name in names
                    if isinstance(context.get(name), Field) and context[name].status == "unknown"
                ),
                None,
            )
            if unresolved is not None:
                name, unresolved_field = unresolved
                needs_verification.append(NeedsVerification(rule=rule, field_name=name, reason=unresolved_field.reason))
                continue

            if rule.evaluate(flat):
                violations.append(Violation(rule=rule, context=flat, detail=rule.render_detail(flat)))

        return Result(violations=violations, needs_verification=needs_verification)
