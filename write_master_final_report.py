from __future__ import annotations

import csv
import json
import re
import shutil
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable


getcontext().prec = 28

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "Liquidity-Index-Research-"

MASTER_REPORT_PATH = PROJECT_ROOT / "MASTER_FINAL_REPORT.md"
REPORT_DATA_PATH = PROJECT_ROOT / "report_data.json"
LEADERBOARD_PATH = PROJECT_ROOT / "results" / "leaderboard.csv"
METRICS_REGISTRY_PATH = PROJECT_ROOT / "results" / "metrics_registry.csv"
DM_TESTS_PATH = PROJECT_ROOT / "validation" / "dm_tests.csv"
WALKFORWARD_PATH = PROJECT_ROOT / "validation" / "walkforward_full.csv"
TEMPORAL_STABILITY_PATH = PROJECT_ROOT / "validation" / "temporal_stability.csv"
ARIMAX_EVALUATION_PATH = PROJECT_ROOT / "validation" / "arimax_evaluation.md"
OVERFIT_ANALYSIS_PATH = PROJECT_ROOT / "validation" / "overfit_analysis.md"
ARTIFACT_INVENTORY_PATH = PROJECT_ROOT / "artifact_inventory.txt"
TEMPORAL_HEATMAP_SOURCE = PROJECT_ROOT / "validation" / "plots" / "temporal_heatmap.png"
TEMPORAL_HEATMAP_TARGET = PROJECT_ROOT / "temporal_heatmap.png"

MAIN_SECTION_SPLIT_RE = re.compile(r"(?m)^## 5\.")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def decimal_from_row(row: dict[str, str], key: str) -> Decimal:
    return Decimal(row[key])


def strip_after_section_five(report_text: str) -> str:
    match = MAIN_SECTION_SPLIT_RE.search(report_text)
    if not match:
        return report_text.rstrip() + "\n\n"
    return report_text[: match.start()].rstrip() + "\n\n"


def escape_md(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    return text


def make_markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_markdown_divider(line: str) -> bool:
    cells = split_markdown_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell)


def parse_markdown_tables(text: str) -> list[list[dict[str, str]]]:
    lines = [line.rstrip() for line in text.splitlines()]
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            current.append(line.strip())
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    tables: list[list[dict[str, str]]] = []
    for block in blocks:
        if len(block) < 2 or not is_markdown_divider(block[1]):
            continue
        headers = split_markdown_row(block[0])
        rows: list[dict[str, str]] = []
        for line in block[2:]:
            cells = split_markdown_row(line)
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            if len(cells) > len(headers):
                cells = cells[: len(headers)]
            rows.append(dict(zip(headers, cells)))
        tables.append(rows)
    return tables


def parse_markdown_sections(text: str) -> list[dict[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            {
                "level": match.group(1),
                "title": match.group(2).strip(),
                "body": text[start:end].strip(),
            }
        )
    return sections


def parse_inventory(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    metadata: dict[str, str] = {}
    entries: list[dict[str, str]] = []
    current_section = ""
    for raw_line in read_text(path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Generated:"):
            metadata["Generated"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Project root:"):
            metadata["Project Root"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Scope:"):
            metadata["Scope"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Files inventoried:"):
            metadata["Files Inventoried"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("== ") and line.endswith(" =="):
            current_section = line[3:-3].strip()
            continue
        if " | " not in line or line.startswith("Format:"):
            continue
        relative_path, size_bytes, description = line.split(" | ", 2)
        entries.append(
            {
                "section": current_section,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "description": description,
            }
        )
    return metadata, entries


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def assert_registry_alignment(
    leaderboard_rows: list[dict[str, str]],
    registry_rows: list[dict[str, str]],
) -> list[str]:
    registry_by_model = {row["model_name"]: row for row in registry_rows}
    registry_only = [row["model_name"] for row in registry_rows if row["model_name"] not in {item["model_name"] for item in leaderboard_rows}]
    for row in leaderboard_rows:
        registry_row = registry_by_model.get(row["model_name"])
        if registry_row is None:
            raise ValueError(f"Leaderboard model missing from metrics registry: {row['model_name']}")
        for metric in ("MAE", "RMSE", "SMAPE"):
            if Decimal(row[metric]) != Decimal(registry_row[metric]):
                raise ValueError(f"Metric mismatch for {row['model_name']} on {metric}.")
    return registry_only


def parse_overfit_table(path: Path) -> list[dict[str, str]]:
    tables = parse_markdown_tables(read_text(path))
    if not tables:
        raise ValueError("No markdown tables found in overfit_analysis.md")
    return tables[0]


def parse_arimax_tables(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    tables = parse_markdown_tables(read_text(path))
    if len(tables) < 2:
        raise ValueError("Expected two markdown tables in arimax_evaluation.md")
    feature_ranking = tables[0]
    comparison_rows = tables[1]
    return feature_ranking, comparison_rows


def ensure_heatmap_copy() -> None:
    if not TEMPORAL_HEATMAP_SOURCE.exists():
        raise FileNotFoundError(f"Temporal heatmap source not found: {TEMPORAL_HEATMAP_SOURCE}")
    if (
        not TEMPORAL_HEATMAP_TARGET.exists()
        or TEMPORAL_HEATMAP_TARGET.stat().st_size != TEMPORAL_HEATMAP_SOURCE.stat().st_size
    ):
        shutil.copy2(TEMPORAL_HEATMAP_SOURCE, TEMPORAL_HEATMAP_TARGET)


def build_section_five(
    leaderboard_rows: list[dict[str, str]],
    registry_only_models: list[str],
    overfit_rows: list[dict[str, str]],
) -> str:
    top_five = leaderboard_rows[:5]
    top_five_models = ", ".join(f"`{row['model_name']}`" for row in top_five)
    top_five_span = decimal_from_row(top_five[4], "RMSE") - decimal_from_row(top_five[0], "RMSE")

    verdict_by_model = {row["Model"]: row for row in overfit_rows}
    overfit_fail_models = [
        "LightGBMRegressor(n_estimators=200)",
        "XGBoostRegressor(n_estimators=200, learning_rate=0.05)",
        "RandomForestRegressor(n_estimators=200, max_depth=10)",
        "SVR(kernel='rbf', C=10, gamma='scale')",
    ]
    underfit_flag_models = [
        "GRU (64 units, dropout=0.2)",
        "LSTM (64 units, dropout=0.2)",
        "Bidirectional LSTM (64 units, dropout=0.2)",
        "Temporal Transformer",
    ]

    leaderboard_table = make_markdown_table(
        ["Rank", "Model", "RMSE", "MAE", "SMAPE"],
        [
            [row["rank"], row["model_name"], row["RMSE"], row["MAE"], row["SMAPE"]]
            for row in leaderboard_rows
        ],
    )

    overfit_summary = ", ".join(
        f"`{name}` (rank {next(row['rank'] for row in leaderboard_rows if row['model_name'] == name)}, "
        f"RMSE {next(row['RMSE'] for row in leaderboard_rows if row['model_name'] == name)}, "
        f"verdict {verdict_by_model[name]['Verdict']})"
        for name in overfit_fail_models
    )
    underfit_summary = ", ".join(
        f"`{name}` (rank {next(row['rank'] for row in leaderboard_rows if row['model_name'] == name)}, "
        f"RMSE {next(row['RMSE'] for row in leaderboard_rows if row['model_name'] == name)}, "
        f"verdict {verdict_by_model[name]['Verdict']})"
        for name in underfit_flag_models
    )

    registry_note = ""
    if registry_only_models:
        registry_note = (
            " The registry carries one additional post-validation row, "
            + ", ".join(f"`{name}`" for name in registry_only_models)
            + ", which is intentionally excluded from the canonical ranking."
        )

    return "\n".join(
        [
            "## 5. Final Leaderboard & Model Performance",
            "",
            (
                "The official ranking remains the `22`-row canonical leaderboard from `results/leaderboard.csv`, "
                "cross-checked against `results/metrics_registry.csv` to ensure the reported `RMSE`, `MAE`, and "
                f"`SMAPE` values match exactly.{registry_note}"
            ),
            "",
            leaderboard_table,
            "",
            (
                f"The post-validation top-5 cluster is {top_five_models}. From rank `1` to rank `5`, the entire "
                f"cluster spans only `{top_five_span}` RMSE, which is too small to support a strong practical "
                "separation before significance testing."
            ),
            "",
            (
                "The clearest overfitting failures remain the tree/nonlinear rows audited as `FAIL`: "
                f"{overfit_summary}. These models fit the training history aggressively but do not preserve that "
                "advantage on the future holdout."
            ),
            "",
            (
                "The underfitting failure mode is concentrated in the neural stack flagged by the train-vs-test audit: "
                f"{underfit_summary}. In those cases, both train and test `RMSE` exceed `120%` of the matched naive "
                "baseline, so higher model capacity did not even buy strong in-sample fit."
            ),
            "",
            (
                "`CNN-LSTM` and `Attention LSTM` are not formal underfit flags in the audit, but they still finish at "
                "ranks `17` and `21`, far outside the production-relevant cluster. The overall ranking therefore "
                "supports a narrow conclusion: the credible frontier is formed by ensemble, sparse linear, and "
                "classical time-series baselines rather than nonlinear trees or neural architectures."
            ),
            "",
        ]
    )


def build_section_six(dm_rows: list[dict[str, str]], report_data: dict[str, object]) -> str:
    summary = report_data["dm_tests"]["summary"]
    dm_table = make_markdown_table(
        ["Model Pair", "DM Statistic", "p-value", "Significant?"],
        [
            [
                f"{row['model_a']} vs {row['model_b']}",
                row["dm_stat"],
                row["p_value"],
                row["significant"],
            ]
            for row in dm_rows
        ],
    )

    closest_pair = summary["closest_to_significance_pair"]
    return "\n".join(
        [
            "## 6. Statistical Significance (DM Tests)",
            "",
            (
                "Pairwise Diebold-Mariano tests were run across the corrected top-5 cluster using squared-error loss, "
                "a two-sided horizon-1 test, and the Harvey-Leybourne-Newbold small-sample correction."
            ),
            "",
            dm_table,
            "",
            (
                f"All `{summary['pair_count']}` pairwise tests are non-significant at `alpha = {summary['alpha']}`. "
                f"The observed `p`-value range is `{summary['min_p_value']}` to `{summary['max_p_value']}`."
            ),
            "",
            (
                "That result means the cluster formed by `Stacking Ensemble (Linear Meta-Learner)`, "
                "`Simple Average Ensemble (Top-3 ML)`, `Lasso(alpha=0.01)`, `Weighted Ensemble (Top-5 Overall)`, "
                "and `Ridge(alpha=1.0)` is statistically indistinguishable on the saved holdout. In particular, "
                "rank `1` is not distinguishable from rank `2` or rank `3` at the stated threshold."
            ),
            "",
            (
                f"The closest pair to significance is `{closest_pair['model_a']}` vs `{closest_pair['model_b']}` at "
                f"`p = {closest_pair['p_value']}`, which is still comfortably above the rejection threshold. The "
                "correct interpretation of the leaderboard is therefore a top-cluster, not a decisive single winner."
            ),
            "",
        ]
    )


def build_section_seven(
    walkforward_rows: list[dict[str, str]],
    report_data: dict[str, object],
) -> str:
    summary = report_data["walkforward_full"]["summary"]
    skipped_models = ", ".join(f"`{row['model_name']}`" for row in summary["skipped_models"])

    tree_models = {
        "RandomForestRegressor(n_estimators=200, max_depth=10)",
        "LightGBMRegressor(n_estimators=200)",
        "XGBoostRegressor(n_estimators=200, learning_rate=0.05)",
    }
    low_capacity_models = {
        "Weighted Ensemble (Top-5 Overall)",
        "Simple Average Ensemble (Top-3 ML)",
        "Stacking Ensemble (Linear Meta-Learner)",
        "SVR(kernel='linear', C=1.0)",
        "Ridge(alpha=1.0)",
        "LinearRegression",
        "Lasso(alpha=0.01)",
    }
    tree_std_values = [Decimal(row["std_rmse"]) for row in walkforward_rows if row["model"] in tree_models]
    low_capacity_std_values = [Decimal(row["std_rmse"]) for row in walkforward_rows if row["model"] in low_capacity_models]
    min_tree_to_low_ratio = min(tree_std_values) / max(low_capacity_std_values)
    max_tree_to_low_ratio = max(tree_std_values) / min(low_capacity_std_values)
    min_tree_to_low_ratio_text = f"{min_tree_to_low_ratio.quantize(Decimal('0.01'))}"
    max_tree_to_low_ratio_text = f"{max_tree_to_low_ratio.quantize(Decimal('0.01'))}"

    walkforward_table = make_markdown_table(
        ["Model", "Family", "Mean CV RMSE", "Std CV RMSE", "CV Inflation %"],
        [
            [
                row["model"],
                row["family"],
                row["mean_rmse"],
                row["std_rmse"],
                row["cv_inflation_pct"],
            ]
            for row in walkforward_rows
        ],
    )

    return "\n".join(
        [
            "## 7. Walk-Forward Cross-Validation",
            "",
            (
                "The extended walk-forward study covers the `16` refittable canonical models over `5` expanding "
                "folds. The `6` skipped rows are the deep-learning models that were not refit in this backtest: "
                f"{skipped_models}."
            ),
            "",
            walkforward_table,
            "",
            (
                f"The best mean walk-forward RMSE belongs to `{summary['best_mean_rmse_model']}` at "
                f"`{summary['best_mean_rmse']}`, while the lowest fold variance belongs to "
                f"`{summary['lowest_fold_variance_model']}` at `std RMSE {summary['lowest_std_rmse']}`."
            ),
            "",
            (
                f"Linear and ensemble models keep maximum CV inflation below "
                f"`{summary['linear_and_ensemble_max_cv_inflation_pct']}%`, which is consistent with the stable "
                "single-split leaderboard. By contrast, the tree-based rows have `std RMSE` between "
                f"`{summary['tree_based_std_rmse_min']}` and `{summary['tree_based_std_rmse_max']}`, or roughly "
                f"`{min_tree_to_low_ratio_text}x` to `{max_tree_to_low_ratio_text}x` the low-capacity fold variance."
            ),
            "",
            (
                "This repeated-window evidence strengthens the post-validation interpretation rather than weakening "
                "it: the low-capacity linear and ensemble family retains its edge across folds, while tree models are "
                "much more regime-sensitive and therefore less reliable candidates for production deployment."
            ),
            "",
        ]
    )


def build_section_eight(
    temporal_rows: list[dict[str, str]],
    report_data: dict[str, object],
) -> str:
    summary = report_data["temporal_stability"]["summary"]
    stability_table = make_markdown_table(
        [
            "Rank",
            "Model",
            "Family",
            "H2-2022 RMSE",
            "2023-H1 RMSE",
            "2023-H2 RMSE",
            "2024-H1 RMSE",
            "2024-H2 RMSE",
            "RMSE Std",
        ],
        [
            [
                row["stability_rank"],
                row["model"],
                row["family"],
                row["h2_2022_rmse"],
                row["2023h1_rmse"],
                row["2023h2_rmse"],
                row["2024h1_rmse"],
                row["2024h2_rmse"],
                row["rmse_std"],
            ]
            for row in temporal_rows
        ],
    )

    period_mean = summary["period_mean_rmse"]
    return "\n".join(
        [
            "## 8. Temporal Stability & Regime Analysis",
            "",
            (
                "Temporal stability was evaluated by slicing the canonical holdout into five contiguous regimes: "
                "`H2-2022`, `2023-H1`, `2023-H2`, `2024-H1`, and `2024-H2`. The corresponding heatmap shows the "
                "strongest and most persistent degradation in the later sample, especially for tree and deep-learning rows."
            ),
            "",
            "![Temporal Heatmap](temporal_heatmap.png)",
            "",
            stability_table,
            "",
            (
                f"The hardest regime is `2024-H1`, with cross-model mean `RMSE {summary['hardest_period']['mean_rmse']}`. "
                f"All `{summary['hardest_period']['models_with_worst_period_here']}` canonical models post their worst "
                "sub-period error there."
            ),
            "",
            (
                f"The easiest regime is `2023-H1`, with cross-model mean `RMSE {summary['easiest_period']['mean_rmse']}`. "
                f"Period means move from `{period_mean['h2_2022_rmse']}` in `H2-2022` down to "
                f"`{period_mean['2023h1_rmse']}` in `2023-H1`, then up to `{period_mean['2024h1_rmse']}` in `2024-H1` "
                f"before easing to `{period_mean['2024h2_rmse']}` in `2024-H2`."
            ),
            "",
            (
                f"The most stable model is `{summary['most_stable_model']['model']}` with `RMSE std "
                f"{summary['most_stable_model']['rmse_std']}`, while the least stable model is "
                f"`{summary['least_stable_model']['model']}` at `RMSE std {summary['least_stable_model']['rmse_std']}`. "
                f"Family means reinforce the same pattern: tree-model mean `RMSE std` is "
                f"`{summary['tree_model_mean_rmse_std']}`, whereas deep-learning mean `RMSE std` rises to "
                f"`{summary['deep_learning_mean_rmse_std']}`."
            ),
            "",
            (
                "The temporal picture is therefore consistent with the corrected leaderboard and the walk-forward "
                "backtest: simple linear/statistical rows are not only accurate but also comparatively stable across "
                "regimes, while later-period deterioration is much sharper for nonlinear ML and DL models."
            ),
            "",
        ]
    )


def build_section_nine(
    feature_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    report_data: dict[str, object],
) -> str:
    feature_table = make_markdown_table(
        ["Rank", "Feature", "Coefficient", "Absolute Coefficient"],
        [[row["Rank"], row["Feature"], row["Coefficient"], row["Absolute Coefficient"]] for row in feature_rows[:5]],
    )
    comparison_table = make_markdown_table(
        ["Model", "MAE", "RMSE", "SMAPE", "R^2", "DA", "Delta RMSE vs ARIMA"],
        [
            [
                row["Model"],
                row["MAE"],
                row["RMSE"],
                row["SMAPE"],
                row["R^2"],
                row["DA"],
                row["Delta RMSE vs ARIMA"],
            ]
            for row in comparison_rows
        ],
    )

    arima_row = next(row for row in comparison_rows if row["Model"] == "ARIMA(1, 1, 1)")
    arimax_row = next(row for row in comparison_rows if row["Model"] == "ARIMAX(1, 1, 1) + Top-5 Exog")
    arimax_summary = report_data["arimax_evaluation"]["summary"]
    exact_rmse_delta_text = f"{abs(Decimal(str(arimax_summary['rmse_delta_vs_arima'])))}"
    relative_delta_pct_text = f"{abs(Decimal(str(arimax_summary['relative_rmse_delta_pct'])))}"

    return "\n".join(
        [
            "## 9. ARIMAX Exogenous Feature Study",
            "",
            (
                "The post-validation extension tested whether lag-derived exogenous regressors could improve the "
                "classical `ARIMA(1, 1, 1)` baseline without violating the corrected future-only evaluation path. "
                "Feature selection was performed with `Ridge(alpha=1.0)` on the train-side candidate set."
            ),
            "",
            feature_table,
            "",
            comparison_table,
            "",
            (
                f"The selected top-5 exogenous features are `lag_1`, `lag_2`, `rolling_mean_21`, `lag_5`, and `lag_4`. "
                f"Even with those regressors, `ARIMAX(1, 1, 1) + Top-5 Exog` reaches `RMSE {arimax_row['RMSE']}` versus "
                f"`{arima_row['RMSE']}` for the plain `ARIMA(1, 1, 1)` baseline."
            ),
            "",
            (
                f"That is a deterioration of `{exact_rmse_delta_text}` RMSE, or `{relative_delta_pct_text}%` relative to the ARIMA "
                "baseline. The study therefore rejects the hypothesis that these lag-only exogenous variables add "
                "incremental information beyond the autoregressive structure already captured by ARIMA."
            ),
            "",
        ]
    )


def build_section_ten(report_data: dict[str, object]) -> str:
    metric_audit = report_data["metric_audit"]["summary"]
    rectification = report_data["rectification_log"]["summary"]
    best_row = report_data["leaderboard"]["rows"][0]
    near_zero_target = metric_audit["near_zero_target"]
    pre_fix_recovery = rectification["pre_rectification_recovery"]
    return "\n".join(
        [
            "## 10. Conclusions & Recommendations",
            "",
            (
                "The project receives a **CONDITIONAL PASS**. After leakage removal, bias correction, metric auditing, "
                "walk-forward validation, temporal stability analysis, and ARIMAX testing, the core family-level result "
                "still holds: low-capacity linear/statistical methods and their disciplined ensembles are credible, "
                "while tree and deep-learning families are not."
            ),
            "",
            "Recommended production posture:",
            "",
            f"- Primary production model: `{best_row['model_name']}` with `RMSE {best_row['RMSE']}`, `MAE {best_row['MAE']}`, and `SMAPE {best_row['SMAPE']}`.",
            "- Practical interpretation: the top-5 cluster is statistically indistinguishable under the corrected DM tests, so `Lasso(alpha=0.01)`, `Ridge(alpha=1.0)`, `Simple Average Ensemble (Top-3 ML)`, and `Weighted Ensemble (Top-5 Overall)` remain defensible backup choices if operational simplicity or retraining cost dominates.",
            f"- Ranking metrics: prefer `RMSE`, `MAE`, and `SMAPE`; avoid `MAPE` as a primary rank key because the holdout contains one near-zero actual on `{near_zero_target['date']}` with value `{near_zero_target['actual_value']}`, and the audit shows MAPE distortions as large as `5.85067` points.",
            f"- Provenance caveat: the rectification audit recovered pre-fix metrics from legacy CSVs for `{pre_fix_recovery['recovered_from_legacy_csv_rows']}` rows and reconstructed them in memory for `{pre_fix_recovery['reconstructed_in_memory_rows']}` rows, so the corrected conclusions are trustworthy but the historical pre-fix archive is not uniformly disk-native.",
            "- Model-family exclusions: tree models and `SVR(kernel='rbf', C=10, gamma='scale')` remain overfitting failures, while the flagged neural rows remain underfit and temporally unstable enough to exclude from deployment consideration.",
            "- Future work: add genuinely external macro or market-state regressors, test non-lag exogenous features, and explore regime-adaptive or switchable models before revisiting higher-capacity deep-learning approaches.",
            "",
            (
                f"In short, the final recommendation is conservative rather than flashy. `{best_row['model_name']}` is "
                "the official production choice, but the real scientific outcome is that the corrected forecasting "
                "frontier is narrow, reproducible, and dominated by simple structures once the evaluation path is made "
                "fully future-safe."
            ),
            "",
        ]
    )


def build_appendix(metadata: dict[str, str], inventory_rows: list[dict[str, str]]) -> str:
    appendix_table = make_markdown_table(
        ["Section", "Relative Path", "Size (bytes)", "Description"],
        [
            [row["section"], row["relative_path"], row["size_bytes"], row["description"]]
            for row in inventory_rows
        ],
    )
    return "\n".join(
        [
            "## Appendix A. Complete Artifact Inventory",
            "",
            (
                "This appendix is a direct markdown conversion of `artifact_inventory.txt`, preserving the full "
                "project-wide file inventory for standalone review."
            ),
            "",
            f"- Generated: `{metadata['Generated']}`",
            f"- Project root: `{metadata['Project Root']}`",
            f"- Files inventoried: `{metadata['Files Inventoried']}`",
            f"- Scope: {metadata['Scope']}",
            "",
            appendix_table,
            "",
        ]
    )


def build_report_text() -> str:
    report_data = json.loads(read_text(REPORT_DATA_PATH))
    existing_report = read_text(MASTER_REPORT_PATH)
    leaderboard_rows = read_csv_rows(LEADERBOARD_PATH)
    registry_rows = read_csv_rows(METRICS_REGISTRY_PATH)
    dm_rows = read_csv_rows(DM_TESTS_PATH)
    walkforward_rows = read_csv_rows(WALKFORWARD_PATH)
    temporal_rows = read_csv_rows(TEMPORAL_STABILITY_PATH)
    overfit_rows = parse_overfit_table(OVERFIT_ANALYSIS_PATH)
    arimax_feature_rows, arimax_comparison_rows = parse_arimax_tables(ARIMAX_EVALUATION_PATH)
    inventory_metadata, inventory_rows = parse_inventory(ARTIFACT_INVENTORY_PATH)

    if len(leaderboard_rows) != 22:
        raise ValueError(f"Expected 22 leaderboard rows, found {len(leaderboard_rows)}")
    if len(dm_rows) != 10:
        raise ValueError(f"Expected 10 DM rows, found {len(dm_rows)}")
    if len(walkforward_rows) != 16:
        raise ValueError(f"Expected 16 walk-forward rows, found {len(walkforward_rows)}")
    if len(temporal_rows) != 22:
        raise ValueError(f"Expected 22 temporal rows, found {len(temporal_rows)}")
    if inventory_metadata.get("Files Inventoried") is None:
        raise ValueError("Artifact inventory metadata is incomplete.")
    if len(inventory_rows) != int(inventory_metadata["Files Inventoried"]):
        raise ValueError(
            f"Artifact inventory row mismatch: expected {inventory_metadata['Files Inventoried']}, found {len(inventory_rows)}"
        )

    registry_only_models = assert_registry_alignment(leaderboard_rows, registry_rows)
    report_prefix = strip_after_section_five(existing_report)

    sections = [
        build_section_five(leaderboard_rows, registry_only_models, overfit_rows),
        build_section_six(dm_rows, report_data),
        build_section_seven(walkforward_rows, report_data),
        build_section_eight(temporal_rows, report_data),
        build_section_nine(arimax_feature_rows, arimax_comparison_rows, report_data),
        build_section_ten(report_data),
        build_appendix(inventory_metadata, inventory_rows),
    ]

    return report_prefix + "\n".join(sections)


def main() -> None:
    ensure_heatmap_copy()
    report_text = build_report_text()
    MASTER_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Wrote {MASTER_REPORT_PATH} ({count_words(report_text)} words).")
    print(f"Ensured heatmap copy at {TEMPORAL_HEATMAP_TARGET}.")


if __name__ == "__main__":
    main()
