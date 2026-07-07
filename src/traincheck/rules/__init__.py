"""Built-in rules for common GPU training misconfigurations.

Split into two modules that must never mix (see each module's own
docstring for why): `config_coherence` (hand-authored, derivable from the
spec) and `version_incompat` (mined from the web with a provenance gate,
currently empty pending verified candidates).
"""

from traincheck.rules.config_coherence import CONFIG_COHERENCE_RULES
from traincheck.rules.version_incompat import VERSION_INCOMPAT_RULES

BUILTIN_RULES = CONFIG_COHERENCE_RULES + VERSION_INCOMPAT_RULES

__all__ = ["BUILTIN_RULES", "CONFIG_COHERENCE_RULES", "VERSION_INCOMPAT_RULES"]
