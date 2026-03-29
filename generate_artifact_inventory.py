from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path.cwd()
PROJECT_ROOT = WORKSPACE_ROOT / "Liquidity-Index-Research-"
OUTPUT_PATH = PROJECT_ROOT / "artifact_inventory.txt"

TEXT_EXTENSIONS = {
    "",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".txt",
}

DIRECT_OVERRIDES = {
    ".gitignore": "Git ignore rules for generated artifacts, caches, and other local-only files.",
    "FINAL_REPORT.md": "Standalone post-validation final report for the liquidity-index ablation study.",
    "RESEARCH_LOG.md": "Chronological research log capturing EDA outputs, modeling milestones, and validation updates.",
    "run_ablation_study.py": "Top-level orchestration script for running the multi-family liquidity-index ablation study.",
    "Code/Computing_Liquidity Proxies.py": "Python pipeline that computes daily liquidity proxies and combines them into the market liquidity index.",
    "Code/Data_Cleaning.py": "Python script for cleaning and standardizing the underlying NSE daily market data before proxy construction.",
    "Code/eda_liquidity_index.py": "Exploratory analysis script for the final market liquidity index time series.",
    "Code/combined_nse_daily_data.csv": "Merged daily NSE market dataset used as the raw starting point for cleaning and proxy computation.",
    "Code/daily_liquidity_proxies.csv": "Daily panel of computed liquidity proxies before PCA-based index aggregation.",
    "Code/filtered_stocks_2010_2024.csv": "Filtered stock universe covering the study period from 2010 through 2024.",
    "Code/market_liquidity_index.csv": "Final daily market liquidity index time series used as the modeling target.",
    "Code/market_liquidity_index_plot.png": "Time-series plot of the market liquidity index over the full study horizon.",
    "Code/market_liquidity_index_30d_plot.png": "Thirty-day smoothed time-series plot of the market liquidity index.",
    "Code/proxy_correlation_heatmap.png": "Correlation heatmap across the constructed liquidity proxies.",
    "Code/proxy_pca_weights.csv": "PCA component weights used to aggregate the proxy set into a single liquidity index.",
    "Data/ind_nifty500list (1).csv": "Reference CSV of Nifty 500 constituents used to define the equity universe.",
    "artifacts/minmax_scalers.joblib": "Serialized MinMaxScaler objects fitted on training data for model preprocessing.",
    "artifacts/preprocessed_arrays.joblib": "Serialized preprocessed feature and target arrays used by the modeling pipelines.",
    "artifacts/split_info.json": "JSON metadata describing the chronological train-test split and warm-up trimming.",
    "artifacts/standard_scalers.joblib": "Serialized StandardScaler objects fitted on training data for model preprocessing.",
    "results/LEADERBOARD.md": "Markdown leaderboard summarizing the corrected cross-family ranking after rectification.",
    "results/MASTER_LEADERBOARD.csv": "Archived cross-family leaderboard from the broader ablation sweep before post-validation corrections.",
    "results/leaderboard.csv": "Official corrected 22-model leaderboard used in the revised final report.",
    "results/metrics_registry.csv": "Canonical registry of post-rectification metrics and prediction artifact paths for the evaluated models.",
    "results/attention_lstm_architecture.txt": "Text specification of the attention-LSTM network architecture explored in the advanced DL sweep.",
    "results/cnn_lstm_architecture.txt": "Text specification of the CNN-LSTM network architecture explored in the advanced DL sweep.",
    "results/temporal_transformer_architecture.txt": "Text specification of the temporal-transformer architecture explored in the advanced DL sweep.",
    "validation/VALIDATION_REPORT.md": "Comprehensive validation report covering leakage, overfitting, metric audits, and study rectification.",
    "validation/analyze_overfit.py": "Validation script that quantifies train-test generalization gaps and flags overfit or underfit models.",
    "validation/analyze_residuals.py": "Validation script that computes residual diagnostics for the strongest post-rectification models.",
    "validation/arimax_evaluation.md": "Markdown note evaluating the ARIMAX extension against the ARIMA baseline.",
    "validation/arimax_exog.py": "Validation extension script that fits ARIMAX with lagged exogenous liquidity-index features.",
    "validation/audit_leakage.py": "Validation audit script that checks the pipeline for temporal leakage and split-boundary violations.",
    "validation/dm_significance.py": "Script that runs Diebold-Mariano significance tests for the highest-ranked models.",
    "validation/dm_tests.csv": "Pairwise Diebold-Mariano test results for the corrected top-ranked models.",
    "validation/dm_tests.md": "Markdown summary of the Diebold-Mariano significance-test findings.",
    "validation/generate_validation_report.py": "Report generator that assembles the validation report from audited CSV and markdown artifacts.",
    "validation/leakage_audit.md": "Markdown audit detailing temporal leakage findings and pass-fail checks across the pipeline.",
    "validation/metric_audit.md": "Markdown audit explaining metric integrity checks and any corrected edge cases.",
    "validation/metric_audit.py": "Validation script that audits metric calculations and numerical edge cases across artifacts.",
    "validation/overfit_analysis.md": "Markdown analysis summarizing overfitting and underfitting behavior across the model families.",
    "validation/overfit_metrics.csv": "CSV of train-test error gaps, baseline comparisons, and verdicts for each audited model.",
    "validation/rectification_log.md": "Markdown log describing how leakage and calibration issues were fixed and re-evaluated.",
    "validation/rectification_metrics.csv": "CSV comparing key metrics before and after the validation-driven rectification pass.",
    "validation/rectify_issues.py": "Rectification script that rebuilds affected artifacts after the validation audit findings.",
    "validation/residual_diagnostics.csv": "CSV of residual autocorrelation and distribution diagnostics for the top corrected models.",
    "validation/residual_diagnostics.md": "Markdown interpretation of the residual diagnostic results.",
    "validation/smoke_test.py": "Reproducibility smoke-test script that verifies key artifacts and chronological split integrity.",
    "validation/smoke_test_output.txt": "Captured output from the validation smoke-test run.",
    "validation/temporal_stability.csv": "Per-model temporal-regime RMSE table used for the stability analysis.",
    "validation/temporal_stability.md": "Markdown summary of the temporal stability and regime-shift analysis.",
    "validation/temporal_stability.py": "Validation script that scores every model across sub-periods to measure temporal stability.",
    "validation/walkforward_cv.csv": "Initial walk-forward cross-validation results for the audited baseline model subset.",
    "validation/walkforward_cv.md": "Markdown summary of the initial walk-forward cross-validation findings.",
    "validation/walkforward_cv.py": "Validation script that performs the initial expanding-window walk-forward backtest.",
    "validation/walkforward_full.csv": "Extended walk-forward cross-validation results for the refittable model families.",
    "validation/walkforward_full.md": "Markdown summary of the extended walk-forward cross-validation study.",
    "validation/walkforward_full.py": "Validation script that runs the full expanding-window backtest across refittable models.",
    "plots/summary/ablation_mae_chart.png": "Bar chart comparing MAE across the ablation-study model families.",
    "plots/summary/mae_family_boxplot.png": "Boxplot showing the distribution of MAE values across model families.",
    "plots/summary/mae_vs_r2_scatter.png": "Scatter plot comparing MAE against R-squared across evaluated models.",
    "plots/eda/acf_pacf_diff.png": "ACF and PACF diagnostics for the differenced liquidity-index series.",
    "plots/eda/acf_pacf_raw.png": "ACF and PACF diagnostics for the raw liquidity-index series.",
    "plots/eda/distribution_analysis.png": "Distribution analysis plot for the market liquidity index.",
    "plots/eda/liquidity_index_timeseries.png": "EDA time-series visualization of the market liquidity index.",
    "plots/eda/monthly_boxplot.png": "Monthly boxplot highlighting seasonality in the market liquidity index.",
    "plots/eda/seasonal_decomposition_21d.png": "Twenty-one-day seasonal decomposition of the market liquidity index.",
    "plots/eda/eda_summary.json": "JSON summary of key exploratory-data-analysis statistics for the market liquidity index.",
    "plots/leaderboard_comparison.png": "Comparison chart visualizing model performance across the final leaderboard.",
    "results/statistical/model_selection_summary.json": "JSON summary of the statistical-model selection and search process.",
}

MODEL_NAME_OVERRIDES = {
    "arima": "ARIMA(1,1,1)",
    "arimax_111_top5exog": "ARIMAX(1,1,1) with top-5 exogenous lags",
    "attention_bilstm": "Attention BiLSTM",
    "attention_lstm": "Attention LSTM",
    "bidirectional_lstm": "Bidirectional LSTM",
    "bigru": "BiGRU",
    "bilstm": "BiLSTM",
    "cnn_gru": "CNN-GRU",
    "cnn_lstm": "CNN-LSTM",
    "cnn_transformer": "CNN-Transformer",
    "elasticnet": "ElasticNet",
    "ensemble_avg_all": "Average Ensemble (All Models)",
    "ensemble_stack_linear": "Stacking Ensemble (Linear Meta-Learner)",
    "ensemble_stack_ridge": "Stacking Ensemble (Ridge Meta-Learner)",
    "ensemble_stack_xgb": "Stacking Ensemble (XGBoost Meta-Learner)",
    "ensemble_top3": "Top-3 Ensemble",
    "ensemble_top5": "Top-5 Ensemble",
    "ensemble_weighted": "Weighted Ensemble",
    "ets": "ETS (SimpleExpSmoothing)",
    "extra_trees": "ExtraTrees",
    "gradient_boosting": "Gradient Boosting",
    "gru": "GRU",
    "gru_stacked": "Stacked GRU",
    "gru_vanilla": "Vanilla GRU",
    "knn_k5": "KNN(k=5)",
    "lasso_alpha001": "Lasso(alpha=0.01)",
    "lightgbm": "LightGBM",
    "linear_regression": "LinearRegression",
    "lstm": "LSTM",
    "lstm_deep": "Deep LSTM",
    "lstm_dropout20": "LSTM with 20% dropout",
    "lstm_dropout40": "LSTM with 40% dropout",
    "lstm_stacked": "Stacked LSTM",
    "lstm_vanilla": "Vanilla LSTM",
    "moving_average": "Moving Average",
    "moving_average_7d": "Moving Average (7-day)",
    "naive": "Naive/Persistence",
    "random_forest": "RandomForest",
    "ridge_alpha1": "Ridge(alpha=1.0)",
    "ridge_alpha10": "Ridge(alpha=10.0)",
    "sarima": "SARIMA(1,1,1)x(0,0,0,21)",
    "simple_average_top3_ml": "Simple Average Ensemble (Top-3 ML)",
    "stacking_linear_top3_base": "Stacking Ensemble (Linear Meta-Learner)",
    "svr_linear": "SVR(kernel='linear', C=1.0)",
    "svr_rbf": "SVR(kernel='rbf', C=10)",
    "tcn": "TCN",
    "temporal_transformer": "Temporal Transformer",
    "transformer_4h": "Transformer (4 heads)",
    "wavenet_cnn": "WaveNet-CNN",
    "weighted_top5_overall": "Weighted Ensemble (Top-5 Overall)",
    "xgboost": "XGBoost",
}

KNOWN_SUFFIXES = (
    "_predictions",
    "_forecast",
    "_feature_importance",
    "_training",
    "_loss_curve",
    "_comparison_all",
    "_comparison",
    "_residuals",
    "_acf",
    "_metrics",
    "_leaderboard",
    "_search",
)
PYCACHE_SUFFIX_RE = re.compile(r"^(?P<module>.+?)\.cpython-\d+$")


@dataclass(frozen=True)
class ArtifactEntry:
    relative_path: str
    size_bytes: int
    description: str


def normalize_rel_path(path: Path) -> str:
    return path.as_posix()


def humanize_token(token: str) -> str:
    lowered = token.lower()
    special = {
        "acf": "ACF",
        "api": "API",
        "adv": "Advanced",
        "arima": "ARIMA",
        "arimax": "ARIMAX",
        "bilstm": "BiLSTM",
        "bigru": "BiGRU",
        "cnn": "CNN",
        "da": "DA",
        "dl": "DL",
        "eda": "EDA",
        "ets": "ETS",
        "gru": "GRU",
        "json": "JSON",
        "knn": "KNN",
        "lgbm": "LightGBM",
        "lightgbm": "LightGBM",
        "lstm": "LSTM",
        "mae": "MAE",
        "mape": "MAPE",
        "ml": "ML",
        "nse": "NSE",
        "pacf": "PACF",
        "pca": "PCA",
        "pdf": "PDF",
        "r2": "R2",
        "rbf": "RBF",
        "rnn": "RNN",
        "rmse": "RMSE",
        "rp1": "RP1",
        "sarima": "SARIMA",
        "smape": "SMAPE",
        "svr": "SVR",
        "tcn": "TCN",
        "txt": "TXT",
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
    }
    if lowered in special:
        return special[lowered]
    if lowered.isdigit():
        return lowered
    return lowered.capitalize()


def humanize_name(raw_name: str) -> str:
    cleaned = raw_name.replace("(1)", "").replace("'", "")
    cleaned = re.sub(r"\s+", " ", cleaned.replace("-", "_")).strip(" _")
    lowered = cleaned.lower()
    if lowered in MODEL_NAME_OVERRIDES:
        return MODEL_NAME_OVERRIDES[lowered]
    tokens = [token for token in re.split(r"[_\s]+", cleaned) if token]
    return " ".join(humanize_token(token) for token in tokens)


def strip_known_suffixes(stem: str) -> str:
    lowered = stem.lower()
    for suffix in KNOWN_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def extract_heading(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = re.sub(r"^#+\s*", "", stripped)
                stripped = stripped.strip("`*_> ")
                return stripped or None
    except OSError:
        return None
    return None


def pyc_module_name(path: Path) -> str:
    match = PYCACHE_SUFFIX_RE.match(path.stem)
    module_stem = match.group("module") if match else path.stem
    if module_stem == "__init__":
        return "package initializer"
    return humanize_name(module_stem)


def load_prediction_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    candidate_csvs = [
        PROJECT_ROOT / "results" / "metrics_registry.csv",
        PROJECT_ROOT / "results" / "leaderboard.csv",
        PROJECT_ROOT / "results" / "MASTER_LEADERBOARD.csv",
        PROJECT_ROOT / "results" / "ml" / "ml_leaderboard.csv",
        PROJECT_ROOT / "results" / "ensemble" / "ensemble_leaderboard.csv",
        PROJECT_ROOT / "results" / "dl_rnn" / "rnn_leaderboard.csv",
        PROJECT_ROOT / "results" / "dl_advanced" / "adv_dl_leaderboard.csv",
        PROJECT_ROOT / "results" / "statistical" / "statistical_leaderboard.csv",
    ]
    for csv_path in candidate_csvs:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                if "predictions_path" not in reader.fieldnames or "model_name" not in reader.fieldnames:
                    continue
                for row in reader:
                    prediction_path = (row.get("predictions_path") or "").strip().replace("\\", "/")
                    model_name = (row.get("model_name") or "").strip()
                    if prediction_path and model_name:
                        mapping[prediction_path] = model_name
        except OSError:
            continue
    return mapping


def describe_prediction_csv(relative_path: str, path: Path, prediction_names: dict[str, str]) -> str:
    normalized = relative_path.replace("\\", "/")
    model_name = prediction_names.get(normalized)
    if not model_name:
        model_name = humanize_name(strip_known_suffixes(path.stem))
    return f"CSV containing holdout actuals and predictions for {model_name}."


def describe_plot_file(path: Path) -> str:
    stem = path.stem
    lowered = stem.lower()
    if lowered.endswith("_forecast"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG forecast plot comparing actual values against predictions for {model_name}."
    if lowered.endswith("_feature_importance"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG feature-importance chart for {model_name}."
    if lowered.endswith("_training"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG training-history plot for {model_name}."
    if lowered.endswith("_loss_curve"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG loss-curve plot for {model_name}."
    if lowered.endswith("_residuals"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG residual-diagnostic plot for {model_name}."
    if lowered.endswith("_acf"):
        model_name = humanize_name(strip_known_suffixes(stem))
        return f"PNG residual autocorrelation plot for {model_name}."
    if lowered.endswith("_comparison"):
        subject = humanize_name(strip_known_suffixes(stem))
        return f"PNG comparison chart for {subject}."
    if lowered.endswith("_comparison_all"):
        subject = humanize_name(strip_known_suffixes(stem))
        return f"PNG comparison chart covering all {subject} variants."
    return f"PNG figure related to {humanize_name(stem)}."


def describe_validation_file(relative_path: str, path: Path) -> str:
    if relative_path in DIRECT_OVERRIDES:
        return DIRECT_OVERRIDES[relative_path]
    suffix = path.suffix.lower()
    if suffix == ".png":
        return describe_plot_file(path)
    if suffix == ".pyc":
        return f"Compiled Python bytecode cache for the validation module {pyc_module_name(path)}."
    if suffix == ".csv":
        topic = humanize_name(strip_known_suffixes(path.stem))
        return f"CSV artifact produced by the validation workflow for {topic}."
    if suffix == ".md":
        heading = extract_heading(path)
        if heading:
            return f"Markdown validation note titled '{heading}'."
    return f"Validation artifact for {humanize_name(path.stem)}."


def describe_results_file(relative_path: str, path: Path, prediction_names: dict[str, str]) -> str:
    parts = Path(relative_path).parts
    suffix = path.suffix.lower()
    name = path.name.lower()
    if relative_path in DIRECT_OVERRIDES:
        return DIRECT_OVERRIDES[relative_path]
    if "predictions" in parts and suffix == ".csv":
        return describe_prediction_csv(relative_path, path, prediction_names)
    if suffix == ".csv" and name.endswith("_leaderboard.csv"):
        family = humanize_name(path.stem.replace("_leaderboard", ""))
        return f"CSV leaderboard ranking the {family} model variants."
    if suffix == ".csv" and name.endswith("_all_metrics.csv"):
        family = humanize_name(path.stem.replace("_all_metrics", ""))
        return f"CSV metrics export covering the {family} model sweep."
    if suffix == ".json" and name.endswith("_all_metrics.json"):
        family = humanize_name(path.stem.replace("_all_metrics", ""))
        return f"JSON metrics archive covering the {family} model sweep."
    if suffix == ".csv" and "model_search" in name:
        model_name = humanize_name(path.stem.replace("_model_search", ""))
        return f"CSV summarizing hyperparameter-search results for {model_name}."
    if suffix == ".csv" and name.endswith("_predictions.csv"):
        model_name = humanize_name(strip_known_suffixes(path.stem))
        return f"CSV containing predictions exported from the family run for {model_name}."
    if suffix == ".png":
        return describe_plot_file(path)
    if suffix == ".txt" and "architecture" in name:
        model_name = humanize_name(path.stem.replace("_architecture", ""))
        return f"Text summary of the {model_name} architecture."
    if suffix == ".json":
        return f"JSON artifact summarizing {humanize_name(path.stem)}."
    return f"Results artifact for {humanize_name(path.stem)}."


def describe_plot_artifact(relative_path: str, path: Path) -> str:
    if relative_path in DIRECT_OVERRIDES:
        return DIRECT_OVERRIDES[relative_path]
    return describe_plot_file(path)


def describe_code_file(relative_path: str, path: Path) -> str:
    if relative_path in DIRECT_OVERRIDES:
        return DIRECT_OVERRIDES[relative_path]
    suffix = path.suffix.lower()
    if suffix == ".py":
        return f"Python script used during data construction or exploratory analysis for {humanize_name(path.stem)}."
    if suffix == ".csv":
        return f"CSV data artifact used during index construction for {humanize_name(path.stem)}."
    if suffix == ".png":
        return describe_plot_file(path)
    return f"Code-side artifact for {humanize_name(path.stem)}."


def describe_src_file(path: Path) -> str:
    relative_path = normalize_rel_path(path)
    if path.suffix.lower() == ".pyc":
        return f"Compiled Python bytecode cache for the source module {pyc_module_name(path)}."
    if relative_path == "src/preprocessing.py":
        return "Core preprocessing module that builds model-ready features and chronological splits."
    if relative_path.startswith("src/models/") and path.suffix.lower() == ".py":
        family = humanize_name(path.stem)
        return f"Source module implementing the {family} modeling workflow."
    if path.name == "__init__.py":
        return f"Python package marker for the {path.parent.name} source package."
    return f"Source artifact for {humanize_name(path.stem)}."


def describe_utils_file(path: Path) -> str:
    relative_path = normalize_rel_path(path)
    if relative_path == "utils/metrics_tracker.py":
        return "Utility module that records and aggregates evaluation metrics across model runs."
    if path.name == "__init__.py":
        return "Python package marker for the shared utilities package."
    if path.suffix.lower() == ".pyc":
        return f"Compiled Python bytecode cache for the utility module {pyc_module_name(path)}."
    return f"Utility artifact for {humanize_name(path.stem)}."


def describe_papers_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return f"Reference PDF related to {humanize_name(path.stem)}."
    if suffix == ".docx":
        return f"Working Word document related to {humanize_name(path.stem)}."
    if suffix == ".csv":
        return f"CSV data file stored in the papers workspace for {humanize_name(path.stem)}."
    if suffix == ".py":
        return f"Python helper script kept in the papers workspace for {humanize_name(path.stem)}."
    if suffix == ".txt":
        return f"Text notes kept in the papers workspace for {humanize_name(path.stem)}."
    return f"Papers workspace artifact for {humanize_name(path.stem)}."


def describe_json_file(path: Path) -> str:
    heading = extract_heading(path)
    if heading:
        return f"JSON artifact capturing structured output for {heading}."
    return f"JSON artifact for {humanize_name(path.stem)}."


def describe_joblib_file(path: Path) -> str:
    return f"Serialized joblib artifact for {humanize_name(path.stem)}."


def describe_file(relative_path: str, path: Path, prediction_names: dict[str, str]) -> str:
    if relative_path in DIRECT_OVERRIDES:
        return DIRECT_OVERRIDES[relative_path]

    top_level = Path(relative_path).parts[0] if Path(relative_path).parts else relative_path
    suffix = path.suffix.lower()

    if top_level == "validation":
        return describe_validation_file(relative_path, path)
    if top_level == "results":
        return describe_results_file(relative_path, path, prediction_names)
    if top_level == "plots":
        return describe_plot_artifact(relative_path, path)
    if top_level == "Code":
        return describe_code_file(relative_path, path)
    if top_level == "src":
        return describe_src_file(path)
    if top_level == "utils":
        return describe_utils_file(path)
    if top_level == "Papers":
        return describe_papers_file(path)
    if top_level == "artifacts":
        if suffix == ".joblib":
            return describe_joblib_file(path)
        if suffix == ".json":
            return describe_json_file(path)
        return f"Serialized artifact for {humanize_name(path.stem)}."
    if top_level == "Data":
        return f"Source data artifact for {humanize_name(path.stem)}."

    if suffix == ".md":
        heading = extract_heading(path)
        if heading:
            return f"Markdown document titled '{heading}'."
        return f"Markdown document for {humanize_name(path.stem)}."
    if suffix == ".py":
        return f"Python source file for {humanize_name(path.stem)}."
    if suffix == ".csv":
        return f"CSV data artifact for {humanize_name(path.stem)}."
    if suffix == ".png":
        return describe_plot_file(path)
    if suffix == ".json":
        return describe_json_file(path)
    if suffix == ".txt":
        return f"Text artifact for {humanize_name(path.stem)}."
    if suffix == ".pyc":
        return f"Compiled Python bytecode cache for {pyc_module_name(path)}."
    return f"Project artifact for {humanize_name(path.stem)}."


def collect_target_files() -> list[Path]:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root not found: {PROJECT_ROOT}")

    scan_roots = [PROJECT_ROOT]
    workspace_validation = WORKSPACE_ROOT / "validation"
    if workspace_validation.exists() and workspace_validation.is_dir():
        scan_roots.append(workspace_validation)

    seen: dict[Path, Path] = {}
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            if path.resolve() == OUTPUT_PATH.resolve():
                continue
            seen[path.resolve()] = path

    return sorted(
        seen.values(),
        key=lambda item: (
            str(item.relative_to(PROJECT_ROOT if item.is_relative_to(PROJECT_ROOT) else WORKSPACE_ROOT)).lower()
        ),
    )


def to_relative_path(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return normalize_rel_path(path.relative_to(PROJECT_ROOT))
    if path.is_relative_to(WORKSPACE_ROOT):
        return normalize_rel_path(path.relative_to(WORKSPACE_ROOT))
    return normalize_rel_path(path)


def build_entries() -> list[ArtifactEntry]:
    prediction_names = load_prediction_name_map()
    entries: list[ArtifactEntry] = []
    for path in collect_target_files():
        relative_path = to_relative_path(path)
        description = describe_file(relative_path, path, prediction_names)
        entries.append(
            ArtifactEntry(
                relative_path=relative_path,
                size_bytes=path.stat().st_size,
                description=description,
            )
        )
    return sorted(entries, key=lambda entry: entry.relative_path.lower())


def render_inventory(entries: list[ArtifactEntry], inventory_size: int) -> str:
    all_entries = sorted(
        [
            *entries,
            ArtifactEntry(
                relative_path="artifact_inventory.txt",
                size_bytes=inventory_size,
                description="Plain-text inventory generated by generate_artifact_inventory.py that lists every project artifact with a size and one-sentence description.",
            ),
        ],
        key=lambda entry: entry.relative_path.lower(),
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "Artifact Inventory",
        f"Generated: {generated_at}",
        f"Project root: {PROJECT_ROOT.as_posix()}",
        "Scope: recursive scan of Liquidity-Index-Research-/, which already contains validation/; no separate top-level validation/ directory exists in the workspace.",
        f"Files inventoried: {len(all_entries):,}",
        "Format: relative_path | size_bytes | description",
        "",
    ]

    grouped: dict[str, list[ArtifactEntry]] = defaultdict(list)
    for entry in all_entries:
        parts = Path(entry.relative_path).parts
        section = parts[0] if len(parts) > 1 else "<root>"
        grouped[section].append(entry)

    for section in sorted(grouped, key=str.lower):
        section_entries = sorted(grouped[section], key=lambda entry: entry.relative_path.lower())
        lines.append(f"== {section} ({len(section_entries)} files) ==")
        for entry in section_entries:
            lines.append(f"{entry.relative_path} | {entry.size_bytes} | {entry.description}")
        lines.append("")
    return "\n".join(lines)


def write_inventory(entries: list[ArtifactEntry]) -> int:
    size_guess = 0
    while True:
        text = render_inventory(entries, size_guess)
        actual_size = len(text.encode("utf-8"))
        if actual_size == size_guess:
            with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            return actual_size
        size_guess = actual_size


def main() -> None:
    entries = build_entries()
    write_inventory(entries)


if __name__ == "__main__":
    main()
