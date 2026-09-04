#!/usr/bin/env python3
"""Fail-closed identity verifier for the prebuilt PPU SpargeAttention extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


class IdentityError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise IdentityError(f"{name}: got {actual!r}, expected {expected!r}")


def public_version(version: str) -> str:
    return version.split("+", 1)[0]


def selection_key(runtime: dict[str, Any]) -> str:
    version = str(runtime["torch_public_version"]).split(".")
    if len(version) < 2 or not all(part.isdigit() for part in version[:2]):
        raise IdentityError(
            f"cannot derive Torch major.minor from {runtime['torch_public_version']!r}"
        )
    python_tag = str(runtime["python_cache_tag"]).replace("-", "")
    abi = int(bool(runtime["cxx11_abi"]))
    return f"{python_tag}-torch{version[0]}.{version[1]}-cxx11abi{abi}"


def runtime_identity() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on the box
        raise IdentityError(f"cannot import torch: {exc}") from exc
    return {
        "python_cache_tag": sys.implementation.cache_tag,
        "torch_public_version": public_version(torch.__version__),
        "torch_build_version": torch.__version__,
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def actlize_revision(repo: Path) -> str:
    """Read the committed gitlink; box execution does not need a checkout."""
    try:
        gitlink = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD:third_party/actlize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise IdentityError(f"cannot resolve the committed actlize gitlink: {exc}") from exc

    checkout = repo / "third_party/actlize"
    if (checkout / ".git").exists():
        try:
            checkout_head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise IdentityError(f"cannot resolve checked-out actlize revision: {exc}") from exc
        require_equal("checked-out actlize revision", checkout_head, gitlink)
    return gitlink


def safe_source(repo: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise IdentityError(f"unsafe source path in manifest: {relative!r}")
    path = repo.joinpath(*pure.parts)
    if not path.is_file():
        raise IdentityError(f"manifest source is missing: {relative}")
    return path


def verify(
    manifest: dict[str, Any],
    repo: Path,
    artifact: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    require_equal("manifest schema", manifest.get("schema_version"), 1)
    require_equal("artifact filename", artifact.name, manifest.get("artifact"))
    if not artifact.is_file():
        raise IdentityError(f"prebuilt artifact is missing: {artifact}")
    require_equal("artifact size", artifact.stat().st_size, manifest.get("artifact_size"))
    artifact_digest = sha256_file(artifact)
    require_equal("artifact sha256", artifact_digest, manifest.get("artifact_sha256"))

    build = manifest.get("build")
    if not isinstance(build, dict):
        raise IdentityError("manifest build identity is missing")
    for field in ("python_cache_tag", "torch_public_version", "cxx11_abi"):
        require_equal(f"runtime {field}", runtime.get(field), build.get(field))
    require_equal(
        "actlize revision", actlize_revision(repo), build.get("actlize_git_commit")
    )

    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise IdentityError("manifest has no source hash authority")
    verified_sources: dict[str, str] = {}
    for relative, expected in sorted(sources.items()):
        actual = sha256_file(safe_source(repo, relative))
        require_equal(f"source sha256 {relative}", actual, expected)
        verified_sources[relative] = actual

    libraries = manifest.get("required_ppu_runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise IdentityError("manifest has no PPU runtime library contract")
    return {
        "verdict": "PASS",
        "artifact": str(artifact),
        "artifact_sha256": artifact_digest,
        "build_source_git_commit": build.get("source_git_commit"),
        "actlize_git_commit": build.get("actlize_git_commit"),
        "runtime": runtime,
        "source_sha256": verified_sources,
        "resource_evidence": manifest.get("resource_evidence"),
        "required_ppu_runtime_libraries": libraries,
    }


def verify_runtime_libraries(
    manifest: dict[str, Any], runtime_dir: Path
) -> list[str]:
    libraries = manifest.get("required_ppu_runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise IdentityError("manifest has no PPU runtime library contract")
    missing = [name for name in libraries if not (runtime_dir / name).is_file()]
    if missing:
        raise IdentityError(
            f"PPU runtime {runtime_dir} is missing libraries: {missing}"
        )
    return libraries


def expected_red(label: str, callback) -> None:
    try:
        callback()
    except IdentityError as exc:
        print(f"[PPU Sparge prebuilt negative] {label}: EXPECTED_RED/PASS ({exc})")
        return
    raise IdentityError(f"negative control {label!r} unexpectedly passed")


def self_test(manifest: dict[str, Any], repo: Path, artifact: Path) -> None:
    runtime = runtime_identity()
    verify(manifest, repo, artifact, runtime)

    changed_binary = json.loads(json.dumps(manifest))
    changed_binary["artifact_sha256"] = sha256_bytes(artifact.read_bytes() + b"changed")
    expected_red(
        "binary-byte-change",
        lambda: verify(changed_binary, repo, artifact, runtime),
    )

    changed_source = json.loads(json.dumps(manifest))
    first_source = next(iter(changed_source["source_sha256"]))
    changed_source["source_sha256"][first_source] = "0" * 64
    expected_red(
        "source-change",
        lambda: verify(changed_source, repo, artifact, runtime),
    )

    wrong_runtime = dict(runtime)
    wrong_runtime["cxx11_abi"] = not runtime["cxx11_abi"]
    expected_red(
        "runtime-abi-change",
        lambda: verify(manifest, repo, artifact, wrong_runtime),
    )
    print("[PPU Sparge prebuilt negative] PASS: all identity mutations rejected")


def select_prebuilt(
    root: Path, runtime: dict[str, Any]
) -> tuple[str, Path, Path, dict[str, Any]]:
    key = selection_key(runtime)
    directory = root / key
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        available = sorted(
            path.name for path in root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        ) if root.is_dir() else []
        raise IdentityError(
            f"no prebuilt artifact for runtime key {key!r}; available={available}"
        )
    manifest = json.loads(manifest_path.read_text())
    artifact_name = manifest.get("artifact")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise IdentityError(f"manifest {manifest_path} has no artifact filename")
    artifact = directory / artifact_name
    return key, manifest_path, artifact, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prebuilt-root", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--artifact-path-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    try:
        runtime = runtime_identity()
        if args.prebuilt_root:
            key, manifest_path, artifact, manifest = select_prebuilt(
                args.prebuilt_root, runtime
            )
        else:
            if args.artifact is None:
                raise IdentityError("--artifact is required with --manifest")
            key = selection_key(runtime)
            manifest_path = args.manifest
            artifact = args.artifact
            manifest = json.loads(manifest_path.read_text())
        if args.self_test:
            self_test(manifest, repo, artifact)
            if args.prebuilt_root:
                unsupported = dict(runtime)
                unsupported["torch_public_version"] = "99.0.0"
                expected_red(
                    "unsupported-runtime-selection",
                    lambda: select_prebuilt(args.prebuilt_root, unsupported),
                )
        result = verify(manifest, repo, artifact, runtime)
        result["selection_key"] = key
        result["manifest"] = str(manifest_path)
        if args.runtime_dir:
            libraries = verify_runtime_libraries(manifest, args.runtime_dir)
            result["ppu_runtime"] = {
                "directory": str(args.runtime_dir),
                "libraries": libraries,
            }
    except (IdentityError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[PPU Sparge prebuilt] FAIL: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    if args.artifact_path_out:
        args.artifact_path_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_path_out.write_text(str(artifact.resolve()) + "\n")
    print(
        "[PPU Sparge prebuilt] PASS: "
        f"selection={result['selection_key']} "
        f"artifact={result['artifact_sha256']} "
        f"sources={len(result['source_sha256'])} "
        f"build_source={result['build_source_git_commit']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
