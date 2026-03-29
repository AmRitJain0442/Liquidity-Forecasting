from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = ROOT / "Liquidity-Index-Research-" / "MASTER_FINAL_REPORT.md"
REQUIRED_TERMS = (
    "0.1562",
    "0.0938",
    "9.9319",
    "2024-H1",
    "Stacking",
    "walk-forward",
)
LEADERBOARD_HEADER_PATTERN = re.compile(
    r"^\|\s*Rank\s*\|\s*Model\s*\|\s*RMSE\s*\|\s*MAE\s*\|\s*SMAPE\s*\|$"
)
MARKDOWN_SEPARATOR_PATTERN = re.compile(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


@dataclass(frozen=True)
class TermResult:
    term: str
    found: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Spot-check MASTER_FINAL_REPORT.md for required metrics, wording, "
            "and presence of the leaderboard markdown table."
        )
    )
    parser.add_argument(
        "report_path",
        nargs="?",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to the markdown report to inspect.",
    )
    return parser.parse_args()


def read_report(report_path: Path) -> str:
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    return report_path.read_text(encoding="utf-8")


def check_terms(report_text: str) -> list[TermResult]:
    return [TermResult(term=term, found=term in report_text) for term in REQUIRED_TERMS]


def has_leaderboard_table(report_text: str) -> bool:
    lines = report_text.splitlines()
    for index, line in enumerate(lines):
        if not LEADERBOARD_HEADER_PATTERN.match(line.strip()):
            continue

        if index + 2 >= len(lines):
            continue

        separator = lines[index + 1].strip()
        first_data_row = lines[index + 2].strip()
        if not MARKDOWN_SEPARATOR_PATTERN.match(separator):
            continue
        if "|" not in first_data_row or "Stacking Ensemble" not in first_data_row:
            continue
        return True

    return False


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_path).resolve()
    report_text = read_report(report_path)

    term_results = check_terms(report_text)
    leaderboard_found = has_leaderboard_table(report_text)
    overall_pass = all(result.found for result in term_results) and leaderboard_found

    print(f"Report: {report_path}")
    for result in term_results:
        status = "FOUND" if result.found else "MISSING"
        print(f"{result.term}: {status}")

    print(f"leaderboard_table: {'FOUND' if leaderboard_found else 'MISSING'}")
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
