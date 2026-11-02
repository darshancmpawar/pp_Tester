"""
Stub heavy ML deps so planner tests don't need tensorflow / sklearn / joblib.

planner.py only uses ml_core.norm and ml_core.cats_match (pure-python helpers),
but ml_core.py imports tensorflow / sklearn / joblib at top level. We register
permissive stub modules that satisfy `from x import Y` without those deps.
"""
import sys
import types


class _PermissiveModule(types.ModuleType):
    """Module that returns a dummy callable for any unknown attribute."""
    def __getattr__(self, name):
        # don't intercept dunders
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **kw: None})


def _ensure_stub(name: str):
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = _PermissiveModule(name)


for mod in (
    "tensorflow",
    "sklearn",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "joblib",
):
    _ensure_stub(mod)
