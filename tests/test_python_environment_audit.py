from __future__ import annotations

from dataclasses import dataclass

from cluster.check_python_environment import audit_distributions


@dataclass
class FakeDistribution:
    name: str
    version: str
    requires: tuple[str, ...] = ()

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


def test_dependency_audit_accepts_satisfied_active_requirements() -> None:
    report = audit_distributions(
        [
            FakeDistribution(
                "consumer_pkg",
                "2.0",
                (
                    "dependency-pkg>=1.0,<2",
                    "missing-extra; extra == 'plot'",
                    "old-python; python_version < '3'",
                ),
            ),
            FakeDistribution("dependency_pkg", "1.5"),
        ]
    )

    assert report["status"] == "pass"
    assert report["issues"] == []
    assert report["packages"] == [
        {"name": "consumer-pkg", "version": "2.0"},
        {"name": "dependency-pkg", "version": "1.5"},
    ]


def test_dependency_audit_reports_missing_incompatible_and_conflicting_versions() -> None:
    report = audit_distributions(
        [
            FakeDistribution("consumer", "1", ("missing>=1", "dependency>=2")),
            FakeDistribution("dependency", "1.5"),
            FakeDistribution("duplicate", "1"),
            FakeDistribution("duplicate", "2"),
        ]
    )

    assert report["status"] == "fail"
    assert {issue["kind"] for issue in report["issues"]} == {
        "conflicting_installed_versions",
        "incompatible_dependency",
        "missing_dependency",
    }
