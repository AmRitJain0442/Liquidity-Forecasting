from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from random import Random

from write_master_final_report import (
    build_report_text,
    count_words,
    ensure_heatmap_copy,
    parse_markdown_sections,
    parse_markdown_tables,
    read_csv_rows,
    read_text,
)


getcontext().prec = 28

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "Liquidity-Index-Research-"

MASTER_REPORT_PATH = PROJECT_ROOT / "MASTER_FINAL_REPORT.md"
RESEARCH_LOG_PATH = PROJECT_ROOT / "RESEARCH_LOG.md"
LEADERBOARD_PATH = PROJECT_ROOT / "results" / "leaderboard.csv"
DM_TESTS_PATH = PROJECT_ROOT / "validation" / "dm_tests.csv"
WALKFORWARD_PATH = PROJECT_ROOT / "validation" / "walkforward_full.csv"
TEMPORAL_STABILITY_PATH = PROJECT_ROOT / "validation" / "temporal_stability.csv"
ARIMAX_EVALUATION_PATH = PROJECT_ROOT / "validation" / "arimax_evaluation.md"
ARTIFACT_INVENTORY_PATH = PROJECT_ROOT / "artifact_inventory.txt"

EXPECTED_TOP_LEVEL_SECTIONS = [
    "1. Dataset & Problem Statement",
    "2. Model Families & Ablation Design",
    "3. Validation Methodology",
    "4. Data Leakage & Bias Rectification",
    "5. Final Leaderboard & Model Performance",
    "6. Statistical Significance (DM Tests)",
    "7. Walk-Forward Cross-Validation",
    "8. Temporal Stability & Regime Analysis",
    "9. ARIMAX Exogenous Feature Study",
    "10. Conclusions & Recommendations",
    "Appendix A. Complete Artifact Inventory",
]

FINAL_LOG_HEADING = "## Master Final Report Finalization (2026-03-29)"
SKIPPED_WALKFORWARD_NOTE = "The `6` skipped rows are the deep-learning models"
WRITE_REPORT_SCRIPT_PATH = ROOT / "write_master_final_report.py"


@dataclass(frozen=True)
class NumericClaim:
    label: str
    source_path: Path
    report_token: str


@dataclass(frozen=True)
class ValidationResult:
    word_count: int
    section_titles: list[str]
    sampled_claims: list[NumericClaim]
    check_messages: list[str]


def decimal_mean(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("Cannot compute the mean of an empty list.")
    return sum(values, Decimal("0")) / Decimal(len(values))


def quantize_decimal(value: Decimal, places: int) -> str:
    exponent = Decimal("1").scaleb(-places)
    return format(value.quantize(exponent, rounding=ROUND_HALF_UP), "f")


def get_section_bodies(report_text: str) -> tuple[list[str], dict[str, str]]:
    top_level_sections = [
        section for section in parse_markdown_sections(report_text) if section["level"] == "##"
    ]
    section_titles = [section["title"] for section in top_level_sections]
    section_map = {section["title"]: section["body"] for section in top_level_sections}
    return section_titles, section_map


def get_first_table(section_body: str, section_title: str) -> list[dict[str, str]]:
    tables = parse_markdown_tables(section_body)
    if not tables:
        raise AssertionError(f"Section '{section_title}' does not contain a markdown table.")
    return tables[0]


def get_top_level_inventory_count() -> int:
    for raw_line in read_text(ARTIFACT_INVENTORY_PATH).splitlines():
        line = raw_line.strip()
        if line.startswith("Files inventoried:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("Could not find 'Files inventoried' in artifact_inventory.txt")


def build_numeric_claim_groups() -> list[list[NumericClaim]]:
    leaderboard_rows = read_csv_rows(LEADERBOARD_PATH)
    dm_rows = read_csv_rows(DM_TESTS_PATH)
    walkforward_rows = read_csv_rows(WALKFORWARD_PATH)
    temporal_rows = read_csv_rows(TEMPORAL_STABILITY_PATH)
    arimax_tables = parse_markdown_tables(read_text(ARIMAX_EVALUATION_PATH))

    if len(arimax_tables) < 2:
        raise ValueError("Expected two markdown tables in arimax_evaluation.md")

    arimax_comparison_rows = arimax_tables[1]
    arimax_row = next(
        row for row in arimax_comparison_rows if row["Model"] == "ARIMAX(1, 1, 1) + Top-5 Exog"
    )

    hardest_regime_mean = quantize_decimal(
        decimal_mean([Decimal(row["2024h1_rmse"]) for row in temporal_rows]),
        6,
    )

    return [
        [
            NumericClaim("Best-model RMSE", LEADERBOARD_PATH, leaderboard_rows[0]["RMSE"]),
            NumericClaim("Best-model MAE", LEADERBOARD_PATH, leaderboard_rows[0]["MAE"]),
            NumericClaim("Worst-model RMSE", LEADERBOARD_PATH, leaderboard_rows[-1]["RMSE"]),
        ],
        [
            NumericClaim(
                "Lowest DM p-value",
                DM_TESTS_PATH,
                min(dm_rows, key=lambda row: Decimal(row["p_value"]))["p_value"],
            ),
            NumericClaim(
                "Highest DM p-value",
                DM_TESTS_PATH,
                max(dm_rows, key=lambda row: Decimal(row["p_value"]))["p_value"],
            ),
            NumericClaim(
                "Stacking-vs-Weighted DM statistic",
                DM_TESTS_PATH,
                next(
                    row["dm_stat"]
                    for row in dm_rows
                    if row["model_a"] == "Stacking Ensemble (Linear Meta-Learner)"
                    and row["model_b"] == "Weighted Ensemble (Top-5 Overall)"
                ),
            ),
        ],
        [
            NumericClaim(
                "Max walk-forward CV inflation",
                WALKFORWARD_PATH,
                max(walkforward_rows, key=lambda row: Decimal(row["cv_inflation_pct"]))[
                    "cv_inflation_pct"
                ],
            ),
            NumericClaim(
                "Best walk-forward mean RMSE",
                WALKFORWARD_PATH,
                min(walkforward_rows, key=lambda row: Decimal(row["mean_rmse"]))["mean_rmse"],
            ),
            NumericClaim(
                "Lowest walk-forward std RMSE",
                WALKFORWARD_PATH,
                min(walkforward_rows, key=lambda row: Decimal(row["std_rmse"]))["std_rmse"],
            ),
        ],
        [
            NumericClaim(
                "Hardest-regime mean RMSE",
                TEMPORAL_STABILITY_PATH,
                hardest_regime_mean,
            ),
            NumericClaim(
                "Most-stable model RMSE std",
                TEMPORAL_STABILITY_PATH,
                min(temporal_rows, key=lambda row: Decimal(row["rmse_std"]))["rmse_std"],
            ),
            NumericClaim(
                "Least-stable model RMSE std",
                TEMPORAL_STABILITY_PATH,
                max(temporal_rows, key=lambda row: Decimal(row["rmse_std"]))["rmse_std"],
            ),
        ],
        [
            NumericClaim("ARIMAX RMSE", ARIMAX_EVALUATION_PATH, arimax_row["RMSE"]),
            NumericClaim(
                "ARIMAX delta RMSE vs ARIMA",
                ARIMAX_EVALUATION_PATH,
                arimax_row["Delta RMSE vs ARIMA"],
            ),
            NumericClaim("ARIMAX directional accuracy", ARIMAX_EVALUATION_PATH, arimax_row["DA"]),
        ],
    ]


def choose_sampled_claims() -> list[NumericClaim]:
    rng = Random(20260329)
    return [rng.choice(group) for group in build_numeric_claim_groups()]


def validate_report(report_text: str) -> ValidationResult:
    section_titles, section_map = get_section_bodies(report_text)
    if section_titles != EXPECTED_TOP_LEVEL_SECTIONS:
        raise AssertionError(
            "Unexpected top-level section layout.\n"
            f"Expected: {EXPECTED_TOP_LEVEL_SECTIONS}\n"
            f"Found: {section_titles}"
        )
    section_check = (
        "CHECK 1 PASS: Sections 1-10 and Appendix A are present "
        f"({len(section_titles)} top-level headings matched)."
    )

    word_count = count_words(report_text)
    if word_count < 2000:
        raise AssertionError(f"MASTER_FINAL_REPORT.md is too short: {word_count} words")
    word_count_check = (
        "CHECK 2 PASS: Word count requirement satisfied "
        f"({word_count} words >= 2000)."
    )

    leaderboard_rows = read_csv_rows(LEADERBOARD_PATH)
    leaderboard_table = get_first_table(
        section_map["5. Final Leaderboard & Model Performance"],
        "5. Final Leaderboard & Model Performance",
    )
    report_models = [row["Model"] for row in leaderboard_table]
    source_models = [row["model_name"] for row in leaderboard_rows]
    if len(leaderboard_table) != 22:
        raise AssertionError(
            f"Leaderboard table row mismatch: expected 22, found {len(leaderboard_table)}"
        )
    if report_models != source_models:
        raise AssertionError("Leaderboard table models do not match results/leaderboard.csv")
    leaderboard_check = (
        "CHECK 4 PASS: Leaderboard table contains all 22 canonical models "
        "from results/leaderboard.csv in source order."
    )

    dm_table = get_first_table(
        section_map["6. Statistical Significance (DM Tests)"],
        "6. Statistical Significance (DM Tests)",
    )
    if len(dm_table) != 10:
        raise AssertionError(f"DM table row mismatch: expected 10, found {len(dm_table)}")
    dm_check = "CHECK 5 PASS: Diebold-Mariano table contains exactly 10 pairwise rows."

    walkforward_table = get_first_table(
        section_map["7. Walk-Forward Cross-Validation"],
        "7. Walk-Forward Cross-Validation",
    )
    if len(walkforward_table) != 16:
        raise AssertionError(
            f"Walk-forward table row mismatch: expected 16, found {len(walkforward_table)}"
        )
    if SKIPPED_WALKFORWARD_NOTE not in section_map["7. Walk-Forward Cross-Validation"]:
        raise AssertionError("Walk-forward section is missing the note about 6 skipped DL models.")
    walkforward_check = (
        "CHECK 6 PASS: Walk-forward table contains exactly 16 rows and notes "
        "that 6 deep-learning models were skipped."
    )

    appendix_table = get_first_table(
        section_map["Appendix A. Complete Artifact Inventory"],
        "Appendix A. Complete Artifact Inventory",
    )
    expected_inventory_rows = get_top_level_inventory_count()
    if len(appendix_table) != expected_inventory_rows:
        raise AssertionError(
            "Appendix inventory row mismatch: "
            f"expected {expected_inventory_rows}, found {len(appendix_table)}"
        )

    sampled_claims = choose_sampled_claims()
    for claim in sampled_claims:
        if claim.report_token not in report_text:
            raise AssertionError(
                f"Numeric claim '{claim.label}' with value {claim.report_token} "
                f"from {claim.source_path.name} is missing from the report."
            )
    numeric_detail = "; ".join(
        f"{claim.label}={claim.report_token} [{claim.source_path.relative_to(PROJECT_ROOT)}]"
        for claim in sampled_claims
    )
    numeric_check = (
        "CHECK 3 PASS: Cross-checked 5 sampled numeric claims against source artifacts "
        f"({numeric_detail})."
    )

    return ValidationResult(
        word_count=word_count,
        section_titles=section_titles,
        sampled_claims=sampled_claims,
        check_messages=[
            section_check,
            word_count_check,
            numeric_check,
            leaderboard_check,
            dm_check,
            walkforward_check,
        ],
    )


def sync_report_to_sources() -> bool:
    ensure_heatmap_copy()
    regenerated_report = build_report_text()
    current_report = read_text(MASTER_REPORT_PATH)
    if regenerated_report == current_report:
        return False
    MASTER_REPORT_PATH.write_text(regenerated_report, encoding="utf-8")
    return True


def upsert_research_log_entry(result: ValidationResult, report_rewritten: bool) -> bool:
    existing_log = read_text(RESEARCH_LOG_PATH)
    sampled_labels = ", ".join(f"`{claim.label}`" for claim in result.sampled_claims)
    status_line = (
        "- Regenerated the source-backed Sections `5`-`10` and `Appendix A` before the final checks, "
        "then reran the validation pass."
        if report_rewritten
        else "- Rechecked the report in place; no content corrections were required after the final validation pass."
    )

    entry = "\n".join(
        [
            FINAL_LOG_HEADING,
            "",
            "### Scope",
            "- Performed the step-5 final QA pass on `MASTER_FINAL_REPORT.md` and closed the standalone research deliverable.",
            "- Updated the research log with the final documentation checkpoint for the consolidated report.",
            "",
            "### Quality Checks",
            "- Verified that the report contains Sections `1` through `10` plus `Appendix A` as top-level markdown headings.",
            f"- Final word count: `{result.word_count}` words, exceeding the `2000`-word requirement.",
            "- Confirmed the Section 5 leaderboard table contains all `22` canonical models from `results/leaderboard.csv` with no extras or omissions.",
            "- Confirmed the Section 6 Diebold-Mariano table contains exactly `10` pairwise rows from `validation/dm_tests.csv`.",
            "- Confirmed the Section 7 walk-forward table contains exactly `16` rows and retains the note that `6` deep-learning models were skipped.",
            (
                "- Cross-checked five sampled numeric claims against source artifacts from "
                "`results/leaderboard.csv`, `validation/dm_tests.csv`, "
                "`validation/walkforward_full.csv`, `validation/temporal_stability.csv`, and "
                f"`validation/arimax_evaluation.md`: {sampled_labels}."
            ),
            status_line,
            "",
            "### Saved Artifacts",
            "- `MASTER_FINAL_REPORT.md`",
            "- `RESEARCH_LOG.md`",
            "- `finalize_master_report.py`",
            "",
        ]
    )

    pattern = re.compile(
        rf"\n{re.escape(FINAL_LOG_HEADING)}\n.*?\Z",
        re.DOTALL,
    )
    if pattern.search(existing_log):
        updated_log = pattern.sub(f"\n{entry}", existing_log.rstrip()) + "\n"
    else:
        updated_log = existing_log.rstrip() + "\n\n" + entry

    if updated_log == existing_log:
        return False

    RESEARCH_LOG_PATH.write_text(updated_log, encoding="utf-8")
    return True


def print_confirmation(result: ValidationResult) -> None:
    for message in result.check_messages:
        print(message)
    print(f"FINAL_WORD_COUNT: {result.word_count}")
    print("SECTION_LIST:")
    for title in result.section_titles:
        print(f"- {title}")
    print("NUMERIC_CLAIMS_CHECKED:")
    for claim in result.sampled_claims:
        relative_path = claim.source_path.relative_to(PROJECT_ROOT)
        print(f"- {claim.label}: {claim.report_token} [{relative_path}]")


def main() -> None:
    if not WRITE_REPORT_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            "write_master_final_report.py is missing at the project root; "
            "inline helpers are required before finalization can proceed."
        )
    report_rewritten = sync_report_to_sources()
    validation_result = validate_report(read_text(MASTER_REPORT_PATH))
    upsert_research_log_entry(validation_result, report_rewritten)
    print_confirmation(validation_result)


if __name__ == "__main__":
    main()
