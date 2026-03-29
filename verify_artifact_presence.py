from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MIN_MASTER_REPORT_BYTES = 70 * 1024


@dataclass(frozen=True)
class ArtifactCheck:
    label: str
    relative_path: str
    min_size_bytes: int
    expectation: str


@dataclass(frozen=True)
class ArtifactResult:
    path: Path
    size_bytes: int
    status: str


REQUIRED_ARTIFACTS: tuple[ArtifactCheck, ...] = (
    ArtifactCheck(
        label="MASTER_FINAL_REPORT.md",
        relative_path="Liquidity-Index-Research-/MASTER_FINAL_REPORT.md",
        min_size_bytes=MIN_MASTER_REPORT_BYTES,
        expectation=f">= {MIN_MASTER_REPORT_BYTES} bytes",
    ),
    ArtifactCheck(
        label="report_data.json",
        relative_path="Liquidity-Index-Research-/report_data.json",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="artifact_inventory.txt",
        relative_path="Liquidity-Index-Research-/artifact_inventory.txt",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="leaderboard.csv",
        relative_path="Liquidity-Index-Research-/results/leaderboard.csv",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="walkforward_full.csv",
        relative_path="Liquidity-Index-Research-/validation/walkforward_full.csv",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="temporal_stability.csv",
        relative_path="Liquidity-Index-Research-/validation/temporal_stability.csv",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="metrics_registry.csv",
        relative_path="Liquidity-Index-Research-/results/metrics_registry.csv",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
    ArtifactCheck(
        label="temporal_heatmap.png",
        relative_path="Liquidity-Index-Research-/temporal_heatmap.png",
        min_size_bytes=1,
        expectation="non-zero size",
    ),
)


def evaluate_artifact(base_dir: Path, artifact: ArtifactCheck) -> ArtifactResult:
    artifact_path = base_dir / artifact.relative_path
    if not artifact_path.exists():
        return ArtifactResult(
            path=artifact_path,
            size_bytes=0,
            status="FAIL",
        )

    size_bytes = artifact_path.stat().st_size
    if size_bytes < artifact.min_size_bytes:
        return ArtifactResult(
            path=artifact_path,
            size_bytes=size_bytes,
            status="FAIL",
        )

    return ArtifactResult(
        path=artifact_path,
        size_bytes=size_bytes,
        status="PASS",
    )


def render_table(rows: Iterable[tuple[str, int, str, str]]) -> str:
    materialized_rows = list(rows)
    headers = ("File", "Size(bytes)", "Status", "Expectation")
    widths = [
        max(len(headers[0]), *(len(row[0]) for row in materialized_rows)),
        max(len(headers[1]), *(len(str(row[1])) for row in materialized_rows)),
        max(len(headers[2]), *(len(row[2]) for row in materialized_rows)),
        max(len(headers[3]), *(len(row[3]) for row in materialized_rows)),
    ]

    def fmt(row: tuple[str, int | str, str, str]) -> str:
        return (
            f"{str(row[0]).ljust(widths[0])}  "
            f"{str(row[1]).rjust(widths[1])}  "
            f"{str(row[2]).ljust(widths[2])}  "
            f"{str(row[3]).ljust(widths[3])}"
        )

    separator = (
        f"{'-' * widths[0]}  "
        f"{'-' * widths[1]}  "
        f"{'-' * widths[2]}  "
        f"{'-' * widths[3]}"
    )

    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in materialized_rows)
    return "\n".join(lines)


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    evaluated_rows = [
        (artifact, evaluate_artifact(base_dir, artifact)) for artifact in REQUIRED_ARTIFACTS
    ]

    table_rows = [
        (
            artifact.relative_path,
            result.size_bytes,
            result.status,
            artifact.expectation,
        )
        for artifact, result in evaluated_rows
    ]
    print(render_table(table_rows))

    failed_results = [
        result for _, result in evaluated_rows if result.status == "FAIL"
    ]
    missing_count = sum(1 for result in failed_results if not result.path.exists())
    undersized_count = len(failed_results) - missing_count

    if not failed_results:
        print("Overall: PASS")
        return 0

    if missing_count and not undersized_count:
        summary = f"Overall: FAIL - {missing_count} artifact(s) missing"
    elif undersized_count and not missing_count:
        summary = f"Overall: FAIL - {undersized_count} artifact(s) below size expectation"
    else:
        summary = (
            "Overall: FAIL - "
            f"{missing_count} missing, {undersized_count} below size expectation"
        )
    print(summary)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
