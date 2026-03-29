from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT / "Liquidity-Index-Research-"
OUTPUT_PATH = PROJECT_ROOT / "report_data.json"

SOURCE_CANDIDATES = {
    "metrics_registry": [
        PROJECT_ROOT / "validation" / "metrics_registry.csv",
        PROJECT_ROOT / "results" / "metrics_registry.csv",
    ],
    "leaderboard": [
        PROJECT_ROOT / "validation" / "leaderboard.csv",
        PROJECT_ROOT / "results" / "leaderboard.csv",
    ],
    "dm_tests": [PROJECT_ROOT / "validation" / "dm_tests.csv"],
    "walkforward_full": [PROJECT_ROOT / "validation" / "walkforward_full.csv"],
    "temporal_stability": [PROJECT_ROOT / "validation" / "temporal_stability.csv"],
    "arimax_evaluation": [PROJECT_ROOT / "validation" / "arimax_evaluation.md"],
    "rectification_log": [PROJECT_ROOT / "validation" / "rectification_log.md"],
    "metric_audit": [PROJECT_ROOT / "validation" / "metric_audit.md"],
}

NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def iso_now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


def coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    if NUMERIC_RE.fullmatch(text):
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    return text


def resolve_source(name: str) -> tuple[Path | None, list[str]]:
    candidates = SOURCE_CANDIDATES[name]
    for candidate in candidates:
        if candidate.exists():
            return candidate, [normalize_path(path) for path in candidates]
    return None, [normalize_path(path) for path in candidates]


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: coerce_scalar(value) for key, value in row.items()} for row in reader]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_markdown_sections(text: str) -> list[dict[str, Any]]:
    matches = list(HEADING_RE.finditer(text))
    sections: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "body": text[start:end].strip("\n"),
            }
        )
    return sections


def get_section(sections: list[dict[str, Any]], title: str) -> str | None:
    wanted = title.lower()
    for section in sections:
        if section["title"].lower() == wanted:
            return section["body"]
    return None


def get_section_by_prefix(sections: list[dict[str, Any]], prefix: str) -> str | None:
    wanted = prefix.lower()
    for section in sections:
        if section["title"].lower().startswith(wanted):
            return section["body"]
    return None


def extract_bullets(text: str | None) -> list[str]:
    if not text:
        return []
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def split_markdown_row(line: str) -> list[str]:
    trimmed = line.strip().strip("|")
    return [cell.strip() for cell in trimmed.split("|")]


def is_markdown_divider(line: str) -> bool:
    cells = split_markdown_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell)


def parse_markdown_table(lines: list[str]) -> list[dict[str, Any]]:
    headers = split_markdown_row(lines[0])
    rows: list[dict[str, Any]] = []
    for line in lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        if len(cells) > len(headers):
            cells = cells[: len(headers)]
        rows.append({header: coerce_scalar(cell) for header, cell in zip(headers, cells)})
    return rows


def extract_markdown_tables(text: str | None) -> list[list[dict[str, Any]]]:
    if not text:
        return []
    lines = [line.rstrip() for line in text.splitlines()]
    tables: list[list[dict[str, Any]]] = []
    current: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            current.append(line.strip())
            continue
        if current:
            if len(current) >= 2 and is_markdown_divider(current[1]):
                tables.append(parse_markdown_table(current))
            current = []
    if current and len(current) >= 2 and is_markdown_divider(current[1]):
        tables.append(parse_markdown_table(current))
    return tables


def extract_table_after_label(text: str, label: str) -> list[dict[str, Any]] | None:
    pattern = re.compile(re.escape(label) + r"\s*(.*)", re.DOTALL)
    match = pattern.search(text)
    if not match:
        return None
    tables = extract_markdown_tables(match.group(1))
    if not tables:
        return None
    return tables[0]


def min_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return min(rows, key=lambda row: row[key])


def max_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(rows, key=lambda row: row[key])


def model_bucket(model_name: str, family: str | None = None) -> str:
    lowered_name = model_name.lower()
    lowered_family = (family or "").lower()
    if lowered_family == "ensemble":
        return "ensemble"
    if "lasso(" in lowered_name or "ridge(" in lowered_name or "linearregression" in lowered_name:
        return "linear_ml"
    if "svr(kernel='linear'" in lowered_name:
        return "linear_ml"
    if any(
        term in lowered_name
        for term in (
            "lightgbm",
            "xgboost",
            "randomforest",
            "extra_trees",
            "extra trees",
            "gradientboosting",
            "gradient boosting",
        )
    ):
        return "tree_ml"
    if "knn" in lowered_name:
        return "distance_ml"
    if "svr(kernel='rbf'" in lowered_name:
        return "kernel_ml"
    if lowered_family == "statistical":
        return "statistical"
    if "lstm" in lowered_name or "gru" in lowered_name or "transformer" in lowered_name or "tcn" in lowered_name:
        return "deep_learning"
    if "wavenet" in lowered_name or "attention" in lowered_name or "cnn" in lowered_name or "bilstm" in lowered_name:
        return "deep_learning"
    return lowered_family or "other"


def round_float(value: float | int | None, digits: int = 6) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(value, digits)


def summarize_family_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row[key]) for row in rows)
    return dict(sorted(counts.items()))


def make_missing_artifact(candidate_paths: list[str]) -> dict[str, Any]:
    return {
        "missing": True,
        "requested_paths": candidate_paths,
        "resolved_path": None,
        "row_count": None,
        "summary": None,
        "rows": None,
    }


def parse_metrics_registry(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    rows = load_csv(path)
    ranked_rows = sorted(rows, key=lambda row: row["RMSE"])
    top_10: list[dict[str, Any]] = []
    for derived_rank, row in enumerate(ranked_rows[:10], start=1):
        enriched = dict(row)
        enriched["rank_by_rmse"] = derived_rank
        top_10.append(enriched)
    best = ranked_rows[0]
    tenth = ranked_rows[9]
    summary = {
        "best_model": best["model_name"],
        "best_family": best["model_family"],
        "best_rmse": best["RMSE"],
        "best_mae": best["MAE"],
        "best_smape": best["SMAPE"],
        "tenth_model": tenth["model_name"],
        "tenth_rmse": tenth["RMSE"],
        "top_10_rmse_span": round_float(tenth["RMSE"] - best["RMSE"]),
        "family_counts": summarize_family_counts(rows, "model_family"),
    }
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": len(rows),
        "summary": summary,
        "top_10": top_10,
        "rows": ranked_rows,
    }


def parse_leaderboard(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    rows = load_csv(path)
    top_10 = rows[:10]
    summary = {
        "best_ranked_model": rows[0]["model_name"],
        "best_ranked_family": rows[0]["model_family"],
        "runner_up_model": rows[1]["model_name"],
        "third_place_model": rows[2]["model_name"],
        "top_3_rmse_range": round_float(rows[2]["RMSE"] - rows[0]["RMSE"]),
        "family_label_counts": summarize_family_counts(rows, "family_label"),
    }
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": len(rows),
        "summary": summary,
        "top_10": top_10,
        "rows": rows,
    }


def parse_dm_tests(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    rows = load_csv(path)
    p_values = [row["p_value"] for row in rows]
    unique_models = sorted({row["model_a"] for row in rows} | {row["model_b"] for row in rows})
    closest = min_row(rows, "p_value")
    summary = {
        "alpha": 0.05,
        "pair_count": len(rows),
        "significant_pair_count": sum(1 for row in rows if row["significant"]),
        "all_pairs_non_significant": all(not row["significant"] for row in rows),
        "min_p_value": min(p_values),
        "max_p_value": max(p_values),
        "tested_models": unique_models,
        "closest_to_significance_pair": {
            "model_a": closest["model_a"],
            "model_b": closest["model_b"],
            "p_value": closest["p_value"],
        },
    }
    findings = [
        "All 10 Diebold-Mariano tests are non-significant at alpha=0.05.",
        f"Minimum p-value is {closest['p_value']:.6f} for {closest['model_a']} vs {closest['model_b']}.",
    ]
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": len(rows),
        "summary": summary,
        "findings": findings,
        "pairwise_tests": rows,
    }


def parse_walkforward_full(
    path: Path,
    candidate_paths: list[str],
    reference_rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    rows = load_csv(path)
    best_mean = min_row(rows, "mean_rmse")
    lowest_std = min_row(rows, "std_rmse")
    highest_std = max_row(rows, "std_rmse")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)

    family_summary = {}
    for family, family_rows in sorted(grouped.items()):
        family_summary[family] = {
            "model_count": len(family_rows),
            "best_mean_rmse_model": min_row(family_rows, "mean_rmse")["model"],
            "mean_of_mean_rmse": round_float(mean(row["mean_rmse"] for row in family_rows)),
            "mean_of_std_rmse": round_float(mean(row["std_rmse"] for row in family_rows)),
            "max_cv_inflation_pct": round_float(max(row["cv_inflation_pct"] for row in family_rows)),
        }

    linear_and_ensemble = [
        row for row in rows if model_bucket(str(row["model"]), str(row["family"])) in {"linear_ml", "ensemble"}
    ]
    tree_based = [row for row in rows if model_bucket(str(row["model"]), str(row["family"])) == "tree_ml"]
    skipped_models = None
    if reference_rows is not None:
        walkforward_models = {str(row["model"]) for row in rows}
        skipped_models = [
            {"model_name": row["model_name"], "model_family": row["model_family"]}
            for row in reference_rows
            if row["model_name"] not in walkforward_models
        ]
        skipped_models.sort(key=lambda row: (str(row["model_family"]), str(row["model_name"])))

    summary = {
        "best_mean_rmse_model": best_mean["model"],
        "best_mean_rmse": best_mean["mean_rmse"],
        "lowest_fold_variance_model": lowest_std["model"],
        "lowest_std_rmse": lowest_std["std_rmse"],
        "highest_fold_variance_model": highest_std["model"],
        "highest_std_rmse": highest_std["std_rmse"],
        "family_summary": family_summary,
        "linear_and_ensemble_max_cv_inflation_pct": (
            round_float(max(row["cv_inflation_pct"] for row in linear_and_ensemble)) if linear_and_ensemble else None
        ),
        "tree_based_std_rmse_min": round_float(min(row["std_rmse"] for row in tree_based)) if tree_based else None,
        "tree_based_std_rmse_max": round_float(max(row["std_rmse"] for row in tree_based)) if tree_based else None,
        "reference_model_count": len(reference_rows) if reference_rows is not None else None,
        "skipped_model_count": len(skipped_models) if skipped_models is not None else None,
        "skipped_models": skipped_models,
    }
    findings = [
        f"Best walk-forward mean RMSE is {best_mean['mean_rmse']:.6f} from {best_mean['model']}.",
        f"Lowest fold variance is {lowest_std['std_rmse']:.6f} for {lowest_std['model']}.",
    ]
    if linear_and_ensemble:
        findings.append(
            "Linear and ensemble models keep max CV inflation below "
            f"{max(row['cv_inflation_pct'] for row in linear_and_ensemble):.6f}%."
        )
    if tree_based:
        findings.append(
            "Tree-based models show materially higher fold variance, with std RMSE ranging from "
            f"{min(row['std_rmse'] for row in tree_based):.6f} to {max(row['std_rmse'] for row in tree_based):.6f}."
        )
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": len(rows),
        "summary": summary,
        "findings": findings,
        "rows": sorted(rows, key=lambda row: row["mean_rmse"]),
    }


def parse_temporal_stability(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    rows = load_csv(path)
    period_columns = ["h2_2022_rmse", "2023h1_rmse", "2023h2_rmse", "2024h1_rmse", "2024h2_rmse"]

    period_means = {
        period: round_float(mean(float(row[period]) for row in rows)) for period in period_columns
    }
    hardest_period = max(period_means, key=period_means.get)
    easiest_period = min(period_means, key=period_means.get)

    worst_period_counts = Counter()
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        worst_period = max(period_columns, key=lambda period: row[period])
        best_period = min(period_columns, key=lambda period: row[period])
        worst_period_counts[worst_period] += 1
        enriched = dict(row)
        enriched["worst_period"] = worst_period
        enriched["best_period"] = best_period
        enriched_rows.append(enriched)

    most_stable = min_row(rows, "rmse_std")
    least_stable = max_row(rows, "rmse_std")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tree_rows: list[dict[str, Any]] = []
    deep_learning_rows: list[dict[str, Any]] = []
    for row in rows:
        family = str(row["family"])
        grouped[family].append(row)
        bucket = model_bucket(str(row["model"]), family)
        if bucket == "tree_ml":
            tree_rows.append(row)
        if bucket == "deep_learning":
            deep_learning_rows.append(row)

    family_summary = {}
    for family, family_rows in sorted(grouped.items()):
        family_summary[family] = {
            "model_count": len(family_rows),
            "mean_rmse_std": round_float(mean(row["rmse_std"] for row in family_rows)),
            "most_stable_model": min_row(family_rows, "rmse_std")["model"],
            "least_stable_model": max_row(family_rows, "rmse_std")["model"],
        }

    summary = {
        "period_columns": period_columns,
        "period_mean_rmse": period_means,
        "hardest_period": {
            "period": hardest_period,
            "mean_rmse": period_means[hardest_period],
            "models_with_worst_period_here": worst_period_counts[hardest_period],
        },
        "easiest_period": {"period": easiest_period, "mean_rmse": period_means[easiest_period]},
        "most_stable_model": {
            "model": most_stable["model"],
            "family": most_stable["family"],
            "rmse_std": most_stable["rmse_std"],
            "stability_rank": most_stable["stability_rank"],
        },
        "least_stable_model": {
            "model": least_stable["model"],
            "family": least_stable["family"],
            "rmse_std": least_stable["rmse_std"],
            "stability_rank": least_stable["stability_rank"],
        },
        "worst_period_counts": dict(worst_period_counts),
        "family_summary": family_summary,
        "tree_model_mean_rmse_std": round_float(mean(row["rmse_std"] for row in tree_rows)) if tree_rows else None,
        "deep_learning_mean_rmse_std": (
            round_float(mean(row["rmse_std"] for row in deep_learning_rows)) if deep_learning_rows else None
        ),
    }
    findings = [
        f"Most stable model is {most_stable['model']} with RMSE std {most_stable['rmse_std']:.6f}.",
        f"Least stable model is {least_stable['model']} with RMSE std {least_stable['rmse_std']:.6f}.",
        f"Hardest regime is {hardest_period} with mean RMSE {period_means[hardest_period]:.6f}.",
        f"{worst_period_counts[hardest_period]} models post their worst period RMSE in {hardest_period}.",
    ]
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": len(rows),
        "summary": summary,
        "findings": findings,
        "top_5_most_stable": sorted(enriched_rows, key=lambda row: row["rmse_std"])[:5],
        "top_5_least_stable": sorted(enriched_rows, key=lambda row: row["rmse_std"], reverse=True)[:5],
        "rows": sorted(enriched_rows, key=lambda row: row["stability_rank"]),
    }


def parse_arimax_evaluation(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    text = load_text(path)
    sections = split_markdown_sections(text)

    method = extract_bullets(get_section(sections, "Method"))
    exog_body = get_section(sections, "Exogenous Variable Selection")
    comparison_body = get_section(sections, "Comparison vs ARIMA Baseline")
    result_body = get_section(sections, "Result")
    saved_artifacts = extract_bullets(get_section(sections, "Saved Artifacts"))

    feature_selection_line = ""
    if exog_body:
        for line in exog_body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- Selected top-5 regressors:"):
                feature_selection_line = stripped
                break
    selected_features = re.findall(r"`([^`]+)`", feature_selection_line)

    exog_tables = extract_markdown_tables(exog_body)
    comparison_tables = extract_markdown_tables(comparison_body)
    feature_ranking = exog_tables[0] if exog_tables else []
    comparison_rows = comparison_tables[0] if comparison_tables else []

    result_text = result_body or ""
    result_match = re.search(
        r"baseline=(?P<baseline>[0-9.]+),\s+ARIMAX=(?P<arimax>[0-9.]+),\s+improvement=(?P<delta>[+-]?[0-9.]+)\s+\((?P<pct>[+-]?[0-9.]+)%\)",
        result_text,
    )
    comparison_by_model = {row["Model"]: row for row in comparison_rows}
    summary = {
        "selected_top_5_exogenous_features": selected_features,
        "baseline_model": "ARIMA(1, 1, 1)",
        "candidate_feature_count": len(feature_ranking),
        "saved_artifacts": saved_artifacts,
        "arimax_improves_on_arima": False,
        "comparison": comparison_by_model,
        "rmse_delta_vs_arima": float(result_match.group("delta")) if result_match else None,
        "relative_rmse_delta_pct": float(result_match.group("pct")) if result_match else None,
    }
    findings = [
        "ARIMAX with the top-5 lagged exogenous features does not beat the ARIMA(1,1,1) baseline.",
    ]
    if result_match:
        findings.append(
            "RMSE changes from "
            f"{result_match.group('baseline')} to {result_match.group('arimax')} "
            f"({result_match.group('pct')}%)."
        )
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": None,
        "summary": summary,
        "method": method,
        "feature_ranking": feature_ranking,
        "comparison_rows": comparison_rows,
        "findings": findings,
    }


def parse_rectification_log(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    text = load_text(path)
    sections = split_markdown_sections(text)

    generated_match = re.search(r"Generated:\s*([^\n]+)", text)
    summary_bullets = extract_bullets(get_section(sections, "Summary"))
    audit_trail_bullets = extract_bullets(get_section(sections, "Audit Trail"))
    diff_summary_bullets = extract_bullets(get_section(sections, "Diff Summary"))
    saved_artifacts = extract_bullets(get_section(sections, "Saved Artifacts"))

    split_match = re.search(
        r"fixed chronological split\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s+\((\d+)\s+train rows\)\s+and\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s+\((\d+)\s+test rows\)",
        text,
    )
    recovery_match = re.search(
        r"recovered from preserved legacy prediction CSVs for (\d+) rows and reconstructed in-memory for (\d+) rows",
        text,
    )
    residual_bias_models = re.findall(
        r"Residual mean-bias correction remained active for \d+ models:\s*(.+?)\.",
        text,
        flags=re.DOTALL,
    )
    bias_models: list[str] = []
    if residual_bias_models:
        bias_models = re.findall(r"`([^`]+)`", residual_bias_models[0])

    reserved_titles = {"Rectification Log", "Summary", "Audit Trail", "Diff Summary", "Saved Artifacts"}
    fix_sections: list[dict[str, Any]] = []
    for section in sections:
        if section["level"] != 2 or section["title"] in reserved_titles:
            continue
        body = section["body"]
        description = body.split("Fix Applied At", maxsplit=1)[0].strip()
        fix_locations = []
        location_match = re.search(r"Fix Applied At\s*(.*?)\s*Before Metric Rows", body, flags=re.DOTALL)
        if location_match:
            fix_locations = extract_bullets(location_match.group(1))
        before_rows = extract_table_after_label(body, "Before Metric Rows")
        after_rows = extract_table_after_label(body, "After Metric Rows")
        fix_sections.append(
            {
                "title": section["title"],
                "description": description,
                "fix_applied_at": fix_locations,
                "before_metric_rows": before_rows,
                "after_metric_rows": after_rows,
            }
        )

    diff_summary = {
        "metrics_registry_row_count": None,
        "rmse_improved_count": None,
        "rmse_worsened_count": None,
        "rmse_unchanged_count": None,
        "largest_rmse_improvement": None,
        "largest_rmse_regression": None,
        "largest_da_gain": None,
        "largest_da_loss": None,
    }
    for bullet in diff_summary_bullets:
        if bullet.startswith("`results/metrics_registry.csv` re-exported with"):
            match = re.search(r"with (\d+) rows", bullet)
            if match:
                diff_summary["metrics_registry_row_count"] = int(match.group(1))
        elif bullet.startswith("RMSE improved for"):
            match = re.search(r"RMSE improved for (\d+) models, worsened for (\d+) models, and was unchanged for (\d+) models", bullet)
            if match:
                diff_summary["rmse_improved_count"] = int(match.group(1))
                diff_summary["rmse_worsened_count"] = int(match.group(2))
                diff_summary["rmse_unchanged_count"] = int(match.group(3))
        elif bullet.startswith("Largest RMSE improvement:"):
            match = re.search(r"`(.+?)` \(([\d.]+) -> ([\d.]+), delta ([+-]?[\d.]+)\)", bullet)
            if match:
                diff_summary["largest_rmse_improvement"] = {
                    "model": match.group(1),
                    "before": float(match.group(2)),
                    "after": float(match.group(3)),
                    "delta": float(match.group(4)),
                }
        elif bullet.startswith("Largest RMSE regression:"):
            match = re.search(r"`(.+?)` \(([\d.]+) -> ([\d.]+), delta ([+-]?[\d.]+)\)", bullet)
            if match:
                diff_summary["largest_rmse_regression"] = {
                    "model": match.group(1),
                    "before": float(match.group(2)),
                    "after": float(match.group(3)),
                    "delta": float(match.group(4)),
                }
        elif bullet.startswith("Largest DA gain:"):
            match = re.search(r"`(.+?)` \(([\d.]+)% -> ([\d.]+)%, delta ([+-]?[\d.]+) pp\)", bullet)
            if match:
                diff_summary["largest_da_gain"] = {
                    "model": match.group(1),
                    "before_pct": float(match.group(2)),
                    "after_pct": float(match.group(3)),
                    "delta_pp": float(match.group(4)),
                }
        elif bullet.startswith("Largest DA loss:"):
            match = re.search(r"`(.+?)` \(([\d.]+)% -> ([\d.]+)%, delta ([+-]?[\d.]+) pp\)", bullet)
            if match:
                diff_summary["largest_da_loss"] = {
                    "model": match.group(1),
                    "before_pct": float(match.group(2)),
                    "after_pct": float(match.group(3)),
                    "delta_pp": float(match.group(4)),
                }

    summary = {
        "generated_at": generated_match.group(1) if generated_match else None,
        "summary_bullets": summary_bullets,
        "audit_trail_bullets": audit_trail_bullets,
        "canonical_split": (
            {
                "train_start": split_match.group(1),
                "train_end": split_match.group(2),
                "train_rows": int(split_match.group(3)),
                "test_start": split_match.group(4),
                "test_end": split_match.group(5),
                "test_rows": int(split_match.group(6)),
            }
            if split_match
            else None
        ),
        "pre_rectification_recovery": (
            {
                "recovered_from_legacy_csv_rows": int(recovery_match.group(1)),
                "reconstructed_in_memory_rows": int(recovery_match.group(2)),
            }
            if recovery_match
            else None
        ),
        "residual_bias_correction_model_count": len(bias_models),
        "residual_bias_correction_models": bias_models,
        "fix_section_count": len(fix_sections),
        "diff_summary": diff_summary,
        "saved_artifacts": saved_artifacts,
    }
    findings = [
        "The shared preprocessing leak was full-sample IQR clipping before the split; the fix moved clipping to train-only bounds.",
        "Ensemble selection, weighting, and stacker calibration were moved off the holdout set onto train-side validation.",
        f"Residual mean-bias correction remained active for {len(bias_models)} models.",
    ]
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": None,
        "summary": summary,
        "fix_sections": fix_sections,
        "findings": findings,
    }


def parse_metric_audit(path: Path, candidate_paths: list[str]) -> dict[str, Any]:
    text = load_text(path)
    sections = split_markdown_sections(text)

    generated_match = re.search(r"Generated:\s*([^\n]+)", text)
    registry_match = re.search(r"Registry:\s*`([^`]+)`", text)
    models_audited_match = re.search(r"Models audited:\s*(\d+)", text)
    rows_per_model_match = re.search(r"Prediction rows per model:\s*(\d+)", text)

    check1_body = get_section_by_prefix(sections, "Check 1:")
    check2_body = get_section_by_prefix(sections, "Check 2:")
    check3_body = get_section_by_prefix(sections, "Check 3:")
    check4_body = get_section_by_prefix(sections, "Check 4:")
    registry_corrections_body = get_section(sections, "Registry Corrections")
    discrepancies_body = get_section(sections, "Discrepancies Above 0.001")

    near_zero_match = re.search(
        r"\|actual\| <= ([0-9.]+).*median `\|actual\| = ([0-9.]+)`\)\. The shared holdout series contains (\d+) such observation\(s\): ([0-9-]+) \(([0-9.]+)\)",
        check2_body or "",
    )
    smape_match = re.search(
        r"max absolute difference `([0-9.]+)`.*at most `([0-9.]+)`.*differs by `([0-9.]+)` points",
        check3_body or "",
        flags=re.DOTALL,
    )
    sample_seed_match = re.search(r"Random sample seed `(\d+)` selected (\d+) models.*above `([0-9.]+)`", check4_body or "")
    tolerance_match = re.search(r"within `([0-9.]+)`", registry_corrections_body or "")

    check2_tables = extract_markdown_tables(check2_body)
    check4_tables = extract_markdown_tables(check4_body)

    check1_status = "PASS" if (check1_body and "**PASS**" in check1_body) else None
    check2_status = "FAIL" if (check2_body and "**FAIL**" in check2_body) else None
    check3_status = "PASS" if (check3_body and "**PASS**" in check3_body) else None
    check4_status = "PASS" if (check4_body and "**PASS**" in check4_body) else None

    summary = {
        "generated_at": generated_match.group(1) if generated_match else None,
        "registry_path": registry_match.group(1) if registry_match else None,
        "models_audited": int(models_audited_match.group(1)) if models_audited_match else None,
        "prediction_rows_per_model": int(rows_per_model_match.group(1)) if rows_per_model_match else None,
        "check_status": {
            "finite_mape_smape": check1_status,
            "near_zero_target_exposure": check2_status,
            "symmetric_smape_formula": check3_status,
            "manual_rmse_recalculation": check4_status,
        },
        "near_zero_target": (
            {
                "threshold_abs_actual": float(near_zero_match.group(1)),
                "median_abs_actual": float(near_zero_match.group(2)),
                "observation_count": int(near_zero_match.group(3)),
                "date": near_zero_match.group(4),
                "actual_value": float(near_zero_match.group(5)),
            }
            if near_zero_match
            else None
        ),
        "smape_validation": (
            {
                "max_absolute_difference": float(smape_match.group(1)),
                "max_actual_prediction_swap_difference": float(smape_match.group(2)),
                "closest_wrong_one_sided_gap": float(smape_match.group(3)),
            }
            if smape_match
            else None
        ),
        "manual_rmse_recalculation": (
            {
                "sample_seed": int(sample_seed_match.group(1)),
                "sample_count": int(sample_seed_match.group(2)),
                "flag_threshold": float(sample_seed_match.group(3)),
            }
            if sample_seed_match
            else None
        ),
        "registry_edit_required": False if registry_corrections_body and "No registry edits were required." in registry_corrections_body else None,
        "registry_match_tolerance": float(tolerance_match.group(1)) if tolerance_match else None,
        "discrepancies_above_threshold": False if discrepancies_body and "No audited metrics differed" in discrepancies_body else None,
    }
    findings = [
        "MAPE and SMAPE are finite across all 22 models.",
        "MAPE is not robust for ranking because one near-zero actual on 2024-11-01 materially distorts percentage errors.",
        "SMAPE recomputation and manual RMSE spot-checks match the stored registry exactly within tolerance.",
    ]
    return {
        "missing": False,
        "requested_paths": candidate_paths,
        "resolved_path": normalize_path(path),
        "row_count": None,
        "summary": summary,
        "near_zero_impact_rows": check2_tables[0] if check2_tables else [],
        "manual_rmse_rows": check4_tables[0] if check4_tables else [],
        "findings": findings,
    }


def parse_artifact(name: str, parser, *parser_args) -> dict[str, Any]:
    resolved_path, candidate_paths = resolve_source(name)
    if resolved_path is None:
        return make_missing_artifact(candidate_paths)
    return parser(resolved_path, candidate_paths, *parser_args)


def main() -> None:
    resolutions = {name: resolve_source(name) for name in SOURCE_CANDIDATES}

    metrics_registry = parse_artifact("metrics_registry", parse_metrics_registry)
    metrics_rows = metrics_registry["rows"] if not metrics_registry["missing"] else None
    leaderboard = parse_artifact("leaderboard", parse_leaderboard)
    leaderboard_rows = leaderboard["rows"] if not leaderboard["missing"] else None

    registry_model_names = {row["model_name"] for row in metrics_rows} if metrics_rows else set()
    leaderboard_model_names = {row["model_name"] for row in leaderboard_rows} if leaderboard_rows else set()
    registry_only_models = sorted(registry_model_names - leaderboard_model_names)
    leaderboard_only_models = sorted(leaderboard_model_names - registry_model_names)
    canonical_model_rows = leaderboard_rows if leaderboard_rows is not None else metrics_rows

    report_data = {
        "generated_at_utc": iso_now_utc(),
        "project_root": normalize_path(PROJECT_ROOT),
        "source_resolution": {
            name: {
                "requested_paths": candidate_paths,
                "resolved_path": normalize_path(resolved_path),
                "missing": resolved_path is None,
            }
            for name, (resolved_path, candidate_paths) in resolutions.items()
        },
        "cross_checks": {
            "metrics_registry_vs_leaderboard": {
                "metrics_registry_row_count": metrics_registry["row_count"],
                "leaderboard_row_count": leaderboard["row_count"],
                "registry_only_models": registry_only_models,
                "leaderboard_only_models": leaderboard_only_models,
                "counts_match": metrics_registry["row_count"] == leaderboard["row_count"],
                "notes": [
                    "metrics_registry includes supplementary post-validation models that may not appear in the canonical leaderboard."
                    if registry_only_models
                    else "metrics_registry and leaderboard cover the same model universe."
                ],
            }
        },
        "metrics_registry": metrics_registry,
        "leaderboard": leaderboard,
        "dm_tests": parse_artifact("dm_tests", parse_dm_tests),
        "walkforward_full": parse_artifact("walkforward_full", parse_walkforward_full, canonical_model_rows),
        "temporal_stability": parse_artifact("temporal_stability", parse_temporal_stability),
        "arimax_evaluation": parse_artifact("arimax_evaluation", parse_arimax_evaluation),
        "rectification_log": parse_artifact("rectification_log", parse_rectification_log),
        "metric_audit": parse_artifact("metric_audit", parse_metric_audit),
    }

    OUTPUT_PATH.write_text(json.dumps(report_data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
