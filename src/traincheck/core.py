"""Core rule engine for traincheck."""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from traincheck.ir import Field

class Severity(Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"

@dataclass
class Rule:
    id: str
    severity: Severity
    condition: str
    message: str
    fix_suggestion: Optional[str] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the rule's condition against the given context.
        Returns True if the rule is violated (i.e., the condition is true).
        """
        try:
            return bool(eval(self.condition, {"__builtins__": {}}, context))
        except Exception:
            return False

@dataclass
class Violation:
    rule: Rule
    context: Dict[str, Any]

@dataclass
class NeedsVerification:
    rule: Rule
    field_name: str
    reason: str

@dataclass
class Result:
    violations: List[Violation] = field(default_factory=list)
    needs_verification: List[NeedsVerification] = field(default_factory=list)

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.WARN]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

def _condition_names(condition: str) -> Set[str]:
    """Names a rule condition reads, e.g. {"nodes", "gpu_type"} for
    "nodes > 32 and gpu_type == 'A100'".
    """
    tree = ast.parse(condition, mode="eval")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

class RuleEngine:
    def __init__(self):
        self._rules: List[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)

    def check(self, context: Dict[str, Any]) -> Result:
        """Run every registered rule against context.

        Context values may be plain values or `Field`s. A rule whose
        condition reads a `Field` with status "unknown" is not evaluated -
        we don't know the value, so we can't safely conclude pass or fail -
        and instead is reported via `Result.needs_verification`.
        """
        violations = []
        needs_verification = []
        flat = {
            name: value.value if isinstance(value, Field) else value
            for name, value in context.items()
        }

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
                needs_verification.append(
                    NeedsVerification(rule=rule, field_name=name, reason=unresolved_field.reason)
                )
                continue

            if rule.evaluate(flat):
                violations.append(Violation(rule=rule, context=flat))

        return Result(violations=violations, needs_verification=needs_verification)
