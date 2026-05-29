"""Simple disk cache for expensive deterministic functions."""
import functools
import inspect
from pathlib import Path

from .io import hash_key, load_pickle, save_pickle


def disk_cache(cache_dir):
    """Decorator: cache a function's outputs on disk by argument hash."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            try:
                sig = inspect.signature(fn)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                key_d = {
                    k: repr(v)[:500]
                    for k, v in bound.arguments.items()
                    if k not in {"self", "cls"}
                }
                key = f"{fn.__name__}_{hash_key(key_d)}"
            except Exception:
                key = f"{fn.__name__}_{hash_key({'a': repr(args)[:500], 'k': repr(kwargs)[:500]})}"
            cf = cache_dir / f"{key}.pkl"
            if cf.exists():
                try:
                    return load_pickle(cf)
                except Exception:
                    pass  # fall through and recompute
            out = fn(*args, **kwargs)
            save_pickle(out, cf)
            return out

        return wrap

    return deco
