"""Non-mutating dependency audit for pip-less locked Python environments.

``uv``-created environments need not contain the ``pip`` module.  This helper
uses installed distribution metadata plus PEP 508 requirement evaluation to
provide the consistency check and deterministic package manifest needed by the
cluster provenance harness without changing the environment.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def audit_distributions(
    distributions: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return installed packages and unsatisfied active requirements."""

    available = list(
        metadata.distributions() if distributions is None else distributions
    )
    packages: list[dict[str, str]] = []
    installed: dict[str, list[str]] = {}
    issues: list[dict[str, str]] = []
    requirements: list[tuple[str, str]] = []

    for distribution in available:
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip():
            issues.append(
                {"kind": "invalid_distribution", "detail": "missing package name"}
            )
            continue
        name = canonicalize_name(raw_name)
        try:
            version = str(Version(raw_version))
        except InvalidVersion:
            issues.append(
                {
                    "kind": "invalid_version",
                    "package": name,
                    "detail": str(raw_version),
                }
            )
            continue
        packages.append({"name": name, "version": version})
        installed.setdefault(name, []).append(version)
        for raw_requirement in distribution.requires or ():
            requirements.append((name, raw_requirement))

    packages.sort(key=lambda item: (item["name"], item["version"]))
    for name, versions in sorted(installed.items()):
        unique_versions = sorted(set(versions))
        if len(unique_versions) > 1:
            issues.append(
                {
                    "kind": "conflicting_installed_versions",
                    "package": name,
                    "detail": ",".join(unique_versions),
                }
            )

    marker_environment = default_environment()
    marker_environment["extra"] = ""
    for owner, raw_requirement in sorted(requirements):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            issues.append(
                {
                    "kind": "invalid_requirement",
                    "package": owner,
                    "detail": raw_requirement,
                }
            )
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(
            marker_environment
        ):
            continue
        dependency = canonicalize_name(requirement.name)
        versions = installed.get(dependency)
        if not versions:
            issues.append(
                {
                    "kind": "missing_dependency",
                    "package": owner,
                    "detail": str(requirement),
                }
            )
            continue
        if requirement.specifier and not any(
            Version(version) in requirement.specifier for version in versions
        ):
            issues.append(
                {
                    "kind": "incompatible_dependency",
                    "package": owner,
                    "detail": f"{requirement}; installed={','.join(sorted(versions))}",
                }
            )

    issues.sort(
        key=lambda item: (item["kind"], item.get("package", ""), item["detail"])
    )
    return {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "python_executable": os.path.realpath(os.sys.executable),
        "package_count": len(packages),
        "packages": packages,
        "issues": issues,
    }


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--packages-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_distributions()
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report is not None:
        _write_atomic(args.report, encoded)
    if args.packages_out is not None:
        package_lines = "".join(
            f"{package['name']}=={package['version']}\n"
            for package in report["packages"]
        )
        _write_atomic(args.packages_out, package_lines)
    print(encoded, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_distributions", "build_parser", "main"]
