"""Tests for velocity.archive — sweep output → RO-Crate reproducibility bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner
from velocity.archive import (
    PROCESS_RUN_CRATE_PROFILE,
    RO_CRATE_CONTEXT,
    RO_CRATE_PROFILE,
    build_ro_crate_metadata,
    create_archive,
)
from velocity.cli import app


def _make_fake_sweep(tmp_path: Path) -> Path:
    """A minimal sweep output dir matching velocity.sweep.run_sweep's layout."""
    out = tmp_path / "20260601T100000Z-sweep"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(_manifest()))
    (out / "comparison.json").write_text(json.dumps({"runs": []}))
    (out / "comparison.md").write_text("# Sweep\n")
    run = out / "fedavg-baseline"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"name": "fedavg-baseline", "seed": 0}))
    (run / "rounds.csv").write_text("round,global_loss\n1,0.5\n")
    (run / "summary.json").write_text(json.dumps({"final_loss": 0.5}))
    return out


def _manifest() -> dict:
    return {
        "vfl_version": "0.1.2",
        "timestamp": "2026-06-01T10:00:00+00:00",
        "host": {"system": "Linux", "release": "6.6", "python": "3.12.3", "cpu_count": 8},
        "git": {"branch": "main", "commit": "abc123", "dirty": False},
    }


def test_ro_crate_metadata_has_required_structure():
    files = [
        "manifest.json",
        "comparison.json",
        "comparison.md",
        "uv.lock",
        "README.md",
        "fedavg-baseline/config.json",
        "fedavg-baseline/rounds.csv",
    ]
    meta = build_ro_crate_metadata(
        files=files,
        manifest=_manifest(),
        root_name="20260601T100000Z-sweep",
        date_published="2026-06-01",
    )
    assert meta["@context"] == RO_CRATE_CONTEXT
    graph = {e["@id"]: e for e in meta["@graph"]}

    # Metadata file descriptor conforms to RO-Crate 1.1 + Process Run Crate profile.
    desc = graph["ro-crate-metadata.json"]
    assert desc["@type"] == "CreativeWork"
    conforms = {c["@id"] for c in desc["conformsTo"]}
    assert RO_CRATE_PROFILE in conforms
    assert PROCESS_RUN_CRATE_PROFILE in conforms
    assert desc["about"] == {"@id": "./"}

    # Root data entity is a Dataset that conforms to the profile and lists every file.
    root = graph["./"]
    assert root["@type"] == "Dataset"
    assert root["conformsTo"] == {"@id": PROCESS_RUN_CRATE_PROFILE}
    haspart = {p["@id"] for p in root["hasPart"]}
    assert "uv.lock" in haspart
    assert "fedavg-baseline/config.json" in haspart

    # Exactly one CreateAction, wired to a SoftwareApplication carrying the vFL version.
    actions = [e for e in meta["@graph"] if e.get("@type") == "CreateAction"]
    assert len(actions) == 1
    instrument = graph[actions[0]["instrument"]["@id"]]
    assert instrument["@type"] == "SoftwareApplication"
    assert instrument["softwareVersion"] == "0.1.2"

    # Every bundled file is its own data entity in the graph.
    for f in files:
        assert f in graph, f"{f} missing from RO-Crate @graph"


def test_create_archive_bundles_rocrate_zip(tmp_path):
    out = _make_fake_sweep(tmp_path)
    lock = tmp_path / "uv.lock"
    lock.write_text("# lock\n")

    archive = create_archive(out, archive_path=tmp_path / "bundle.crate.zip", lockfile=lock)

    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("ro-crate-metadata.json"))
    # RO-Crate metadata at the zip root + the bundled artifacts, lock, and README.
    assert "ro-crate-metadata.json" in names
    assert "README.md" in names
    assert "uv.lock" in names
    assert "comparison.json" in names
    assert "manifest.json" in names
    assert "fedavg-baseline/config.json" in names

    assert meta["@context"] == RO_CRATE_CONTEXT
    graph_ids = {e["@id"] for e in meta["@graph"]}
    assert "uv.lock" in graph_ids
    assert "README.md" in graph_ids
    assert "fedavg-baseline/config.json" in graph_ids
    # The lockfile and README are not themselves data entities of the metadata file.
    assert "ro-crate-metadata.json" not in {
        p["@id"] for e in meta["@graph"] if e["@id"] == "./" for p in e["hasPart"]
    }


def test_create_archive_without_lockfile_falls_back_to_installed_packages(tmp_path):
    out = _make_fake_sweep(tmp_path)
    archive = create_archive(
        out, archive_path=tmp_path / "b.crate.zip", lockfile=tmp_path / "nope" / "uv.lock"
    )
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        packages = zf.read("installed-packages.txt").decode()
    assert "uv.lock" not in names
    assert "installed-packages.txt" in names
    assert "velocity-fl==" in packages or "velocity_fl==" in packages


def test_create_archive_rejects_non_sweep_dir(tmp_path):
    empty = tmp_path / "not-a-sweep"
    empty.mkdir()
    with pytest.raises(ValueError, match="does not look like a sweep"):
        create_archive(empty, archive_path=tmp_path / "x.zip")


def test_cli_archive_command_writes_zip(tmp_path):
    out = _make_fake_sweep(tmp_path)
    dest = tmp_path / "cli.crate.zip"
    result = CliRunner().invoke(
        app, ["archive", str(out), "-o", str(dest), "--lockfile", str(tmp_path / "missing.lock")]
    )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert str(dest) in result.output
    with zipfile.ZipFile(dest) as zf:
        assert "ro-crate-metadata.json" in zf.namelist()
