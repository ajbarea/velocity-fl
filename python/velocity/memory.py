"""Per-user semantic & procedural memory for vFL.

Transparent, file-based: every researcher's memory lives as markdown files
the researcher can read, edit, or delete at any time. No vector DB, no
service dependency. Mirrors Anthropic's long-running-Claude pattern.

Layout (override root with ``VFL_MEMORY_DIR``):

    ~/.velocity/memory/
      {user_id}/
        MEMORY.md         # index; always loaded into the agent's context
        profile.md        # who they are, research focus, defaults
        style.md          # observed communication preferences
        hypotheses.md     # active research threads
        recent_runs.md    # rolling compacted summary of the last N runs
        recipes.md        # user-taught procedural memory
        preferences.md    # concrete prefs (LaTeX, plot colors, naming)
        .events.jsonl     # append-only ledger of every write (transparency)

Any agent write goes through :func:`write_entry`, which records an event.
Readers can inspect the ledger at any time via :func:`events`.
"""

from __future__ import annotations

import getpass
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / ".velocity" / "memory"

# Files the agent is allowed to auto-write to. Keeps the surface small so a
# misbehaving agent can't scatter arbitrary files into the user's memory dir.
WRITABLE_FILES = frozenset(
    {
        "MEMORY.md",
        "profile.md",
        "style.md",
        "hypotheses.md",
        "recent_runs.md",
        "recipes.md",
        "preferences.md",
    }
)


def memory_root() -> Path:
    return Path(os.environ.get("VFL_MEMORY_DIR", DEFAULT_ROOT))


def default_user_id() -> str:
    return os.environ.get("VFL_USER_ID") or getpass.getuser()


def user_dir(user_id: str) -> Path:
    d = memory_root() / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _events_path(user_id: str) -> Path:
    return user_dir(user_id) / ".events.jsonl"


def _log_event(user_id: str, action: str, file: str, summary: str) -> None:
    event = {
        "ts": datetime.now(UTC).isoformat(),
        "action": action,
        "file": file,
        "summary": summary,
    }
    with _events_path(user_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def list_files(user_id: str) -> list[str]:
    return sorted(
        p.name for p in user_dir(user_id).iterdir() if p.is_file() and not p.name.startswith(".")
    )


def read_entry(user_id: str, file: str) -> str:
    path = user_dir(user_id) / file
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_entry(user_id: str, file: str, content: str, summary: str) -> None:
    """Overwrite a memory file and log the change.

    ``summary`` is a short human-readable description of *why* this change
    happened — shown back to the user when they audit the ledger.
    """
    if file not in WRITABLE_FILES:
        raise ValueError(f"{file!r} is not in the writable memory set")
    path = user_dir(user_id) / file
    path.write_text(content, encoding="utf-8")
    _log_event(user_id, "write", file, summary)


def append_entry(user_id: str, file: str, block: str, summary: str) -> None:
    """Append a block to a memory file (preferred over full rewrites).

    Append-with-ledger minimizes poisoning risk: old content is preserved,
    every addition is traceable.
    """
    if file not in WRITABLE_FILES:
        raise ValueError(f"{file!r} is not in the writable memory set")
    path = user_dir(user_id) / file
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size > 0:
            f.write("\n")
        f.write(block.rstrip() + "\n")
    _log_event(user_id, "append", file, summary)


def forget_entry(user_id: str, file: str, reason: str) -> None:
    path = user_dir(user_id) / file
    if path.exists():
        path.unlink()
    _log_event(user_id, "delete", file, reason)


def compact_entry(user_id: str, file: str, keep_last_n: int = 10) -> int:
    """Bound a memory file by keeping only its last N H2 blocks.

    Treats `## ` (H2) headers as block separators, mirroring how callers
    append per-run summaries to `recent_runs.md`. The text before the first
    H2 is preserved as the file preamble (the bootstrap title or any prose
    the user added). The N most recent blocks are kept verbatim; the older
    ones are replaced by a single compaction marker.

    Returns the number of blocks dropped (0 if the file was already within
    bounds or had no H2 blocks at all). No-op if the file does not exist or
    if it has <= keep_last_n H2 blocks.

    The compaction itself is auditable: writes a `compact` event to the
    user's `.events.jsonl` ledger naming the file and the count dropped.
    Each individual run's full text is NOT preserved post-compaction; if
    you need that history, read `.events.jsonl` (every `append` to this
    file recorded a `summary`) or query `db.recent_runs(user_id, limit=...)`
    for the structured DB snapshot.

    Raises:
        ValueError: if ``file`` is not in :data:`WRITABLE_FILES`.
    """
    if file not in WRITABLE_FILES:
        raise ValueError(f"{file!r} is not in the writable memory set")
    if keep_last_n < 0:
        raise ValueError(f"keep_last_n must be non-negative; got {keep_last_n!r}")
    path = user_dir(user_id) / file
    if not path.exists():
        return 0

    text = path.read_text(encoding="utf-8")
    parts = text.split("\n## ")
    if len(parts) <= 1:
        return 0  # No H2 blocks to compact.

    preamble = parts[0].rstrip()
    blocks = ["## " + b for b in parts[1:]]
    if len(blocks) <= keep_last_n:
        return 0  # Already within bounds.

    dropped = len(blocks) - keep_last_n
    kept_blocks = blocks[-keep_last_n:] if keep_last_n > 0 else []
    today = datetime.now(UTC).date().isoformat()
    plural = "s" if dropped != 1 else ""
    marker = (
        f"## (compacted {dropped} earlier block{plural} on {today})\n\n"
        "_Earlier per-block summaries remain in `.events.jsonl`; "
        "full run snapshots remain queryable via `db.recent_runs`._"
    )
    new_text = preamble + "\n\n" + marker + "\n\n" + "\n".join(kept_blocks)
    path.write_text(new_text, encoding="utf-8")
    _log_event(
        user_id,
        "compact",
        file,
        f"kept last {keep_last_n} blocks; dropped {dropped}",
    )
    return dropped


def events(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    p = _events_path(user_id)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[-limit:]]


_BOOTSTRAPPED: set[str] = set()


def bootstrap(user_id: str) -> None:
    """Create empty scaffolding for a new user so the agent has something to read."""
    if user_id in _BOOTSTRAPPED:
        return
    d = user_dir(user_id)
    if not (d / "MEMORY.md").exists():
        (d / "MEMORY.md").write_text(
            f"# Memory index for {user_id}\n\n"
            "- [Profile](profile.md) — who I am as a researcher\n"
            "- [Style](style.md) — communication preferences\n"
            "- [Hypotheses](hypotheses.md) — active research threads\n"
            "- [Recent runs](recent_runs.md) — compacted summary of my last N experiments\n"
            "- [Recipes](recipes.md) — named workflows I've taught the agent\n"
            "- [Preferences](preferences.md) — concrete prefs (plots, naming, export formats)\n",
            encoding="utf-8",
        )
        _log_event(user_id, "bootstrap", "MEMORY.md", "initial scaffolding")
    for f in (
        "profile.md",
        "style.md",
        "hypotheses.md",
        "recent_runs.md",
        "recipes.md",
        "preferences.md",
    ):
        fp = d / f
        if not fp.exists():
            fp.write_text(
                f"# {f.removesuffix('.md').replace('_', ' ').title()}\n\n_empty_\n",
                encoding="utf-8",
            )
    _BOOTSTRAPPED.add(user_id)
