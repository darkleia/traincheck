"""Core rule engine for traincheck."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
class Result:
    violations: List[Violation] = field(default_factory=list)

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.rule.severity == Severity.WARN]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

class RuleEngine:
    def __init__(self):
        self._rules: List[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)

    def check(self, context: Dict[str, Any]) -> Result:
        violations = []
        for rule in self._rules:
            if rule.evaluate(context):
                violations.append(Violation(rule=rule, context=context))
        return Result(violations=violations)
