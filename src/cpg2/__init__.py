"""CPG-2 real-EHR experiment framework.

The package evaluates tree observability and mechanically checkable behavior.  It
does not assign clinical gold labels to real encounters.
"""

from .executor import execute_partial
from .registry import CriterionRegistry
from .states import CriterionState
from .tree import load_tree, validate_tree

__all__ = [
    "CriterionRegistry",
    "CriterionState",
    "execute_partial",
    "load_tree",
    "validate_tree",
]

__version__ = "0.1.0"
