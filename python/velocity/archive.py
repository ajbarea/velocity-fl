"""velocity.archive — package a sweep output directory into a single-file
reproducibility bundle.

The bundle is an RO-Crate (Process Run Crate profile): the existing sweep
artifacts plus a dependency lock, a how-to-reproduce README, and a
machine-readable ``ro-crate-metadata.json``, zipped into one shareable file a
collaborator or reviewer can re-run from.

The CLI wrapper (`velocity archive`) lives in ``velocity.cli``; this module is
importable for agent / programmatic use, mirroring ``velocity.sweep``.

research(2026-06): RO-Crate is the community standard for packaging a research
artifact with machine-readable provenance; the Process Run Crate profile is the
one for "a tool was executed and produced these outputs" (no formal workflow
entity required). Emitted by hand with stdlib ``json`` — the spec states JSON-LD
tooling is not required to produce a conformant metadata file — so no new
dependency. Sources: researchobject.org/workflow-run-crate/profiles/process_run_crate,
researchobject.org/ro-crate/specification/1.1.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from velocity.sweep import RunSpec, SweepResult

RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"
RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.1"
PROCESS_RUN_CRATE_PROFILE = "https://w3id.org/ro/wfrun/process/0.5"

_SOFTWARE_ID = "#velocity"
_ACTION_ID = "#velocity-sweep-run"
_REPO_URL = "https://github.com/ajbarea/velocity-fl"

# Bundled files whose basename marks them a run input (config) vs an output
# (everything the run produced). Drives the CreateAction object/result split.
_INPUT_BASENAMES = frozenset({"config.json"})


def build_ro_crate_metadata(
    *,
    files: list[str],
    manifest: dict[str, Any],
    root_name: str,
    date_published: str,
) -> dict[str, Any]:
    """Build a Process Run Crate ``ro-crate-metadata.json`` as a dict.

    ``files`` are the bundle-relative paths of every data file in the crate
    (excluding the metadata file itself). ``manifest`` is the sweep-time
    provenance dict (``velocity.sweep.capture_manifest`` shape).
    """
    inputs = [f for f in files if f.rsplit("/", 1)[-1] in _INPUT_BASENAMES]
    outputs = [f for f in files if f not in inputs]

    action: dict[str, Any] = {
        "@id": _ACTION_ID,
        "@type": "CreateAction",
        "name": "velocity sweep",
        "description": "Federated-learning experiment sweep that produced this crate.",
        "instrument": {"@id": _SOFTWARE_ID},
        "object": [{"@id": f} for f in inputs],
        "result": [{"@id": f} for f in outputs],
    }
    if manifest.get("timestamp"):
        action["endTime"] = manifest["timestamp"]

    software: dict[str, Any] = {
        "@id": _SOFTWARE_ID,
        "@type": "SoftwareApplication",
        "name": "Velocity-FL",
        "url": _REPO_URL,
        "softwareVersion": manifest.get("vfl_version", ""),
    }

    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "conformsTo": [
                {"@id": RO_CRATE_PROFILE},
                {"@id": PROCESS_RUN_CRATE_PROFILE},
            ],
            "about": {"@id": "./"},
        },
        {
            "@id": "./",
            "@type": "Dataset",
            "name": root_name,
            "description": "Velocity-FL reproducibility archive (sweep run).",
            "datePublished": date_published,
            "license": {"@id": "https://spdx.org/licenses/MIT"},
            "conformsTo": {"@id": PROCESS_RUN_CRATE_PROFILE},
            "hasPart": [{"@id": f} for f in files],
            "mentions": {"@id": _ACTION_ID},
        },
        software,
        action,
    ]
    graph.extend({"@id": f, "@type": "File", "name": f.rsplit("/", 1)[-1]} for f in files)

    return {"@context": RO_CRATE_CONTEXT, "@graph": graph}


def _looks_like_sweep(out_dir: Path) -> bool:
    """A sweep output dir carries a comparison/manifest or per-run configs."""
    return (
        (out_dir / "comparison.json").is_file()
        or (out_dir / "manifest.json").is_file()
        or any(out_dir.glob("*/config.json"))
    )


def _discover_lockfile(start: Path) -> Path | None:
    """Find ``uv.lock`` by walking up from the sweep dir, then from cwd."""
    for base in (start.resolve(), Path.cwd()):
        for d in (base, *base.parents):
            cand = d / "uv.lock"
            if cand.is_file():
                return cand
    return None


def _installed_packages_text() -> str:
    """`name==version` lines for the active environment (uv.lock fallback)."""
    seen: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name not in seen:
            seen[name] = dist.version
    return "".join(f"{n}=={v}\n" for n, v in sorted(seen.items()))


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    """The sweep-time provenance manifest, or a fresh capture if it predates it."""
    mpath = out_dir / "manifest.json"
    if mpath.is_file():
        return json.loads(mpath.read_text())
    from velocity.sweep import capture_manifest

    return capture_manifest()


def _render_reproduce_readme(
    *, root_name: str, manifest: dict[str, Any], dep_arcname: str, artifact_files: list[str]
) -> str:
    host = manifest.get("host") or {}
    git = manifest.get("git") or {}
    configs = [f for f in artifact_files if f.endswith("/config.json")]
    install = "uv sync --frozen" if dep_arcname == "uv.lock" else f"pip install -r {dep_arcname}"
    dirty = " (dirty working tree)" if git.get("dirty") else ""
    lines = [
        f"# Reproducibility archive — {root_name}",
        "",
        "A Velocity-FL sweep packaged as an RO-Crate (Process Run Crate profile).",
        "`ro-crate-metadata.json` is the machine-readable manifest of this bundle.",
        "",
        "## Provenance",
        "",
        f"- Velocity-FL version: `{manifest.get('vfl_version', '?')}`",
        f"- Python: `{host.get('python', '?')}`",
        f"- git commit: `{git.get('commit') or '?'}`{dirty}",
        "",
        "## Reproduce",
        "",
        f"1. Pin the dependency set from `{dep_arcname}`:",
        "   ```",
        f"   {install}",
        "   ```",
        f"2. Each `<run>/config.json` is a serialized Velocity-FL `RunSpec` "
        f"({len(configs)} run(s)). Re-run programmatically:",
        "   ```python",
        "   import json",
        "   from velocity.sweep import RunSpec, run_sweep",
        "   spec = RunSpec.model_validate(json.load(open('<run>/config.json')))",
        "   run_sweep([spec], out_dir='reproduced')",
        "   ```",
        "   A future `velocity reproduce <archive.zip>` will automate this round-trip.",
        "",
        "## Contents",
        "",
        f"- `{dep_arcname}` — pinned dependency set",
        "- `manifest.json` — provenance (version, host, git)",
        "- `comparison.{json,md}` — sweep results + ranking",
        "- `<run>/{config.json,rounds.csv,summary.json}` — per-run spec + results",
        "",
    ]
    return "\n".join(lines) + "\n"


def create_archive(
    out_dir: Path,
    *,
    archive_path: Path | None = None,
    lockfile: Path | None = None,
) -> Path:
    """Bundle a sweep output directory into a single-file RO-Crate Zip.

    Adds a dependency lock (``uv.lock`` if discoverable, else an
    ``installed-packages.txt`` fallback), a how-to-reproduce ``README.md``, and a
    Process Run Crate ``ro-crate-metadata.json`` at the zip root. Returns the
    written archive path (defaults to ``<out_dir>.crate.zip`` beside the sweep).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} is not a directory")
    if not _looks_like_sweep(out_dir):
        raise ValueError(
            f"{out_dir} does not look like a sweep output "
            "(no comparison.json, manifest.json, or */config.json)"
        )

    archive_path = (
        Path(archive_path) if archive_path else out_dir.parent / f"{out_dir.name}.crate.zip"
    )
    manifest = _load_manifest(out_dir)

    lockfile = Path(lockfile) if lockfile is not None else _discover_lockfile(out_dir)
    if lockfile is not None and lockfile.is_file():
        dep_arcname, dep_bytes = "uv.lock", lockfile.read_bytes()
    else:
        dep_arcname, dep_bytes = "installed-packages.txt", _installed_packages_text().encode()

    artifact_files = sorted(
        str(p.relative_to(out_dir)).replace("\\", "/") for p in out_dir.rglob("*") if p.is_file()
    )
    files = [*artifact_files, dep_arcname, "README.md"]

    readme = _render_reproduce_readme(
        root_name=out_dir.name,
        manifest=manifest,
        dep_arcname=dep_arcname,
        artifact_files=artifact_files,
    )
    metadata = build_ro_crate_metadata(
        files=files,
        manifest=manifest,
        root_name=out_dir.name,
        date_published=datetime.now(UTC).date().isoformat(),
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in artifact_files:
            zf.write(out_dir / rel, rel)
        zf.writestr(dep_arcname, dep_bytes)
        zf.writestr("README.md", readme)
        zf.writestr("ro-crate-metadata.json", json.dumps(metadata, indent=2))
    return archive_path


@dataclass
class ArchiveContents:
    """What `read_archive` recovers from a reproducibility crate."""

    specs: list[RunSpec]
    original: dict[str, Any] | None  # parsed comparison.json (a SweepResult dump), if present


def read_archive(archive_path: Path) -> ArchiveContents:
    """Recover the per-run `RunSpec`s (and original results) from a crate zip.

    Reads the bundled `<run>/config.json` entries and `comparison.json` directly
    from the archive — no extraction to disk needed to inspect or re-run it.
    """
    from velocity.sweep import RunSpec

    archive_path = Path(archive_path)
    specs: list[RunSpec] = []
    original: dict[str, Any] | None = None
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        for name in sorted(n for n in names if n.endswith("/config.json")):
            specs.append(RunSpec.model_validate_json(zf.read(name)))
        if "comparison.json" in names:
            original = json.loads(zf.read("comparison.json"))
    if not specs:
        raise ValueError(f"{archive_path} has no <run>/config.json — not a velocity archive")
    return ArchiveContents(specs=specs, original=original)


def reproduce_archive(archive_path: Path, *, out_dir: Path) -> SweepResult:
    """Re-run an archived sweep from its bundled configs into ``out_dir``.

    A reproduction in the ACM/NISO sense: same configs + code, re-executed. Reuses
    the existing ``run_sweep`` runner (DRY) over the recovered specs.
    """
    from velocity.sweep import run_sweep

    contents = read_archive(archive_path)
    return run_sweep(contents.specs, out_dir=Path(out_dir))


@dataclass
class ResultDiff:
    """One run's archived vs reproduced final loss, and whether they agree."""

    name: str
    original: float | None
    reproduced: float
    ok: bool


def _loss_close(a: float | None, b: float, rel_tol: float) -> bool:
    if a is None:
        return False  # run absent from the original results
    a_nan = isinstance(a, float) and a != a
    b_nan = isinstance(b, float) and b != b
    if a_nan and b_nan:
        return True  # both undefined → not a mismatch (the offline stub's losses are nan)
    if a_nan or b_nan:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=0.0)


def compare_results(
    original: dict[str, Any], reproduced: SweepResult, *, rel_tol: float = 1e-6
) -> list[ResultDiff]:
    """Per-run final-loss agreement within a relative tolerance (nan-safe).

    Tolerance-based, not bit-exact: ML / float aggregation is not bitwise
    deterministic across runs and hardware, so asserting equality would emit false
    failures. ``original`` is a parsed ``comparison.json`` (a ``SweepResult`` dump).
    """
    orig_runs = {r["spec"]["name"]: r for r in original.get("runs", [])}
    diffs: list[ResultDiff] = []
    for run in reproduced.runs:
        name = run.spec.name
        if name in orig_runs:
            raw = orig_runs[name].get("final_loss")
            # pydantic serializes an in-memory nan loss to JSON null, so a present
            # run with null final_loss means nan, not "missing".
            original_loss = float("nan") if raw is None else raw
        else:
            original_loss = None  # run absent from the archived results
        diffs.append(
            ResultDiff(
                name=name,
                original=original_loss,
                reproduced=run.final_loss,
                ok=_loss_close(original_loss, run.final_loss, rel_tol),
            )
        )
    return diffs
