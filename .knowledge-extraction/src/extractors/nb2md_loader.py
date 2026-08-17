"""
Loader for nb2md — the repo's single notebook parser.

`knowledge-base/nb2md.py` is a standalone, stdlib-only script rather than an
installed package, so it is loaded by path. Everything that needs to know what
a notebook contains goes through it, so stub detection and output handling stay
consistent between the rendered notes and the knowledge base.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

NB2MD_RELATIVE_PATH = Path('knowledge-base') / 'nb2md.py'

_cached: ModuleType | None = None


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until a directory containing nb2md.py is found."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / NB2MD_RELATIVE_PATH).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate {NB2MD_RELATIVE_PATH} in any parent of {here}. "
        "The knowledge extractor requires nb2md as its notebook parser."
    )


def load_nb2md() -> ModuleType:
    """Import nb2md as a module. Cached — the file is read once per process."""
    global _cached
    if _cached is not None:
        return _cached

    module_path = find_repo_root() / NB2MD_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location('nb2md', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclasses/pickle inside nb2md can resolve it.
    sys.modules['nb2md'] = module
    spec.loader.exec_module(module)

    _cached = module
    return module
