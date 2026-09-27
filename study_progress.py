"""Terminal progress bars; falls back to a no-op iterator if tqdm is missing."""
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable
