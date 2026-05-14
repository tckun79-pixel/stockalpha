"""Search history — persistent JSON-backed list of recent tickers.

Reads/writes ~/.stockalpha_history.json (stdlib json module only).
"""

import json
import os
from pathlib import Path

_HISTORY_FILE = Path.home() / ".stockalpha_history.json"
_MAX_ENTRIES = 10


def _load() -> list[str]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(_HISTORY_FILE.read_text())
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save(history: list[str]) -> None:
    try:
        _HISTORY_FILE.write_text(json.dumps(history, indent=2))
    except OSError:
        pass


def push(ticker: str) -> list[str]:
    """Prepend *ticker* to history, deduplicate, cap at _MAX_ENTRIES.

    Returns the updated list so callers can immediately render it.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        return _load()

    history = _load()
    # Remove existing occurrence so we can move it to the front
    history = [t for t in history if t != ticker]
    history.insert(0, ticker)
    history = history[:_MAX_ENTRIES]
    _save(history)
    return history


def pop() -> list[str]:
    """Read current history without mutating."""
    return _load()


def clear() -> list[str]:
    """Wipe the history file and return an empty list."""
    _save([])
    return []