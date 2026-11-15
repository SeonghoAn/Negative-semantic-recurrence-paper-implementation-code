from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, t as student_t
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Core analysis utilities

RATES = (0.10, 0.15, 0.20)
MODEL = "sentence-transformers/all-mpnet-base-v2"
TARGET = "negative_Lmax_mean_10_15_20"
ATQ = "AutomaticThought"
COVARIATES = ["n_sentences", "negative_sentence_prop", "age", "sex_M"]
PMC = ["Precuneus", "PCC"]
MTL = ["Anterior PHG", "Posterior PHG", "Hippocampus", "Amygdala"]
NODES = PMC + MTL
EDGES = [f"{p}--{m}" for p in PMC for m in MTL]


def subject_id(value):
    match = re.search(r"(?:sub[-_]?)?(\d+)", str(value).strip(), flags=re.I)
    return f"sub-{int(match.group(1)):03d}" if match else str(value).strip()


def cosine_similarity(embeddings):
    x = np.asarray(embeddings, dtype=float)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return np.clip(x @ x.T, -1, 1)


def similarity_threshold(similarity, rate):
    n = len(similarity)
    if n < 2:
        return np.inf
    values = similarity[np.triu_indices(n, 1)]
    return float(np.quantile(values, 1 - rate))


def longest_diagonal(matrix):
    x = np.asarray(matrix, dtype=bool)
    n = len(x)
    dp = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        if i == 0:
            dp[i, 1:] = x[i, 1:]
        else:
            dp[i, i + 1:] = x[i, i + 1:] * (dp[i - 1, i:-1] + 1)
    best = int(dp.max(initial=0))
    return best if best >= 2 else 0


def mnsrl(similarity, valence, thresholds=None):
    negative = np.asarray(valence, dtype=float) < 0
    mask = np.outer(negative, negative)
    lines = []
    threshold_values = []
    for rate in RATES:
        threshold = similarity_threshold(similarity, rate) if thresholds is None else thresholds[len(lines)]
        rec = similarity >= threshold
        np.fill_diagonal(rec, False)
        lines.append(longest_diagonal(rec & mask))
        threshold_values.append(threshold)
    return float(np.mean(lines)), lines, threshold_values


def shuffled_mnsrl(similarity, negative, thresholds, rng, kind):
    if kind == "sentence_order":
        order = rng.permutation(len(negative))
        sim = similarity[np.ix_(order, order)]
        valence = negative[order]
    else:
        sim = similarity
        valence = rng.permutation(negative)
    return mnsrl(sim, np.where(valence, -1.0, 1.0), thresholds)[0]


def partial_pearson(frame, x, y, covariates=COVARIATES):
    columns = [x, y, *covariates]
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    predictors = [col for col in covariates if data[col].nunique() > 1]
    design = np.column_stack([np.ones(len(data)), data[predictors].to_numpy(dtype=float)])
    xv = data[x].to_numpy(dtype=float)
    yv = data[y].to_numpy(dtype=float)
    xr = xv - design @ np.linalg.lstsq(design, xv, rcond=None)[0]
    yr = yv - design @ np.linalg.lstsq(design, yv, rcond=None)[0]
    r = float(np.corrcoef(xr, yr)[0, 1])
    df = len(data) - len(predictors) - 2
    t = r * np.sqrt(df / (1 - np.clip(r * r, 0, 1 - 1e-12)))
    p = float(2 * stats.t.sf(abs(t), df))
    return r, p, len(data), df


def bootstrap_partial(frame, x, y, covariates=COVARIATES, iterations=5000, seed=42):
    data = frame[[x, y, *covariates]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for b in range(iterations):
        sample = data.iloc[rng.integers(0, len(data), len(data))]
        estimates[b] = partial_pearson(sample, x, y, covariates)[0]
    return np.nanpercentile(estimates, [2.5, 97.5])


def bootstrap_pearson(frame, x, y, iterations=5000, seed=42):
    data = frame[[x, y]].dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    for i in range(iterations):
        block = data.iloc[rng.integers(0, len(data), len(data))]
        estimates[i] = stats.pearsonr(block[x], block[y]).statistic
    return np.nanpercentile(estimates, [2.5, 97.5])


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    return multipletests(p, method="fdr_bh")[1]


def permutation_partial_r(observed, null_values, subjects, table, outcome=ATQ, covariates=COVARIATES):
    selected = table.set_index("subject").loc[subjects]
    design = np.column_stack([np.ones(len(selected)), selected[covariates].to_numpy(dtype=float)])
    y = selected[outcome].to_numpy(dtype=float)
    residualizer = np.eye(len(selected)) - design @ np.linalg.pinv(design)
    yr = residualizer @ y
    xr = residualizer @ np.asarray(null_values, dtype=float)
    r_null = (yr @ xr) / (np.linalg.norm(yr) * np.linalg.norm(xr, axis=0))
    p = (1 + np.sum(np.abs(r_null) >= abs(observed))) / (len(r_null) + 1)
    return float(p), r_null


def signed_svd(edges):
    W = np.array([[edges[f"{p}--{m}"] for m in MTL] for p in PMC], dtype=float)
    u, singular, vt = np.linalg.svd(W, full_matrices=False)
    for j in range(2):
        if u[0, j] < 0:
            u[:, j] *= -1
            vt[j, :] *= -1
    energy = singular**2 / np.sum(singular**2)
    values = {f"node_contrib__mode{j + 1}__{node.replace(' ', '_')}": float(u[i, j] ** 2)
              for j in range(2) for i, node in enumerate(PMC)}
    values.update({f"node_contrib__mode{j + 1}__{node.replace(' ', '_')}": float(vt[j, i] ** 2)
                   for j in range(2) for i, node in enumerate(MTL)})
    values.update({
        "svd_decomp__sigma1": float(singular[0]),
        "svd_decomp__sigma2": float(singular[1]),
        "svd_decomp__mode1_matrix_energy_share": float(energy[0]),
        "svd_decomp__mode2_matrix_energy_share": float(energy[1]),
        "signed_svd__matrix_frobenius_norm": float(np.linalg.norm(W)),
    })
    return values, W, u, singular, vt


def association(frame, feature, target=TARGET, covariates=COVARIATES, iterations=5000, seed=42):
    r, p, n, df = partial_pearson(frame, feature, target, covariates)
    lower, upper = bootstrap_partial(frame, feature, target, covariates, iterations, seed)
    return {"feature": feature, "target": target, "partial_r": r, "p_raw": p,
            "ci95_low": lower, "ci95_high": upper, "n": n, "df": df}


def export(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# Figure 1 plotting utilities (original figure calculations preserved)

DEFAULT_TARGET = "AutomaticThought"
DEFAULT_FEATURE = "observed_Lmax_mean_rrs"
DEFAULT_COVARIATES = ["n_sentences", "negative_sentence_prop", "age", "sex_code"]
BOOTSTRAP_N_DEFAULT = 5000
BOOTSTRAP_SEED_DEFAULT = 20260822


def format_number(value, digits=3):
    if not np.isfinite(value):
        return "NA"
    text = f"{float(value):.{digits}f}"
    return text.replace("-0.", "-.", 1).replace("0.", ".", 1) if text.startswith(("-0.", "0.")) else text

def fig1_significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return "n.s."


def clean_label(x: str) -> str:
    mapping = {
        "n_sentences": "No. of sentences",
        "negative_sentence_prop": "Negative sentence proportion",
        "negative_sentence_count": "No. of negative sentences",
        "observed_Lmax_mean_rrs": "mNSRL",
        "Lmax_excess_mean_rrs": "Null-adjusted mNSRL",
        "AutomaticThought": "ATQ",
        "Rumination": "RRS",
        "age": "Age",
        "sex_code": "Sex",
    }
    return mapping.get(x, x)


def short_axis_label(x: str) -> str:
    mapping = {
        "n_sentences": "Sentence\ncount",
        "negative_sentence_prop": "Negative\nsentence\nproportion",
        "negative_sentence_count": "Negative\nsentence\ncount",
        "observed_Lmax_mean_rrs": "mNSRL",
        "Lmax_excess_mean_rrs": "Null-adjusted\nmNSRL",
    }
    return mapping.get(x, clean_label(x))


def bootstrap_pearson_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_boot: int = BOOTSTRAP_N_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
    ci: float = 95.0,
) -> Tuple[float, float, int]:
    """
    Participant-level percentile bootstrap CI for a simple Pearson correlation.

    Complete cases are selected once for the requested x/y pair. Participants
    are then resampled with replacement, and Pearson r is recomputed for each
    bootstrap sample. Bootstrap samples with zero variance are skipped.
    """
    tmp = df[[x_col, y_col]].copy()
    tmp[x_col] = pd.to_numeric(tmp[x_col], errors="coerce")
    tmp[y_col] = pd.to_numeric(tmp[y_col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    x = tmp[x_col].to_numpy(dtype=float)
    y = tmp[y_col].to_numpy(dtype=float)
    n = len(tmp)

    if n < 4 or n_boot <= 0:
        return np.nan, np.nan, 0

    # Stable feature-specific seed so results do not depend on predictor order.
    feature_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(f"{x_col}|{y_col}"))
    rng = np.random.default_rng((int(seed) + feature_offset) % (2**32 - 1))

    boot_r = np.full(int(n_boot), np.nan, dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        if np.nanstd(xb) <= 0 or np.nanstd(yb) <= 0:
            continue

        boot_r[b] = float(np.corrcoef(xb, yb)[0, 1])

    valid = boot_r[np.isfinite(boot_r)]
    if valid.size == 0:
        return np.nan, np.nan, 0

    alpha = (100.0 - float(ci)) / 2.0
    lo, hi = np.percentile(valid, [alpha, 100.0 - alpha])
    return float(lo), float(hi), int(valid.size)


def _as_2d_covariates(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim == 1:
        cov = cov[:, None]
    return cov


def residualize_vector(y: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    C = _as_2d_covariates(covariates)
    if C.shape[1] == 0:
        return y - np.nanmean(y)
    X = np.column_stack([np.ones(C.shape[0]), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr_with_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray,
) -> Tuple[float, float, int, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    C = _as_2d_covariates(covariates)

    finite = np.isfinite(x) & np.isfinite(y)
    if C.shape[1] > 0:
        finite &= np.all(np.isfinite(C), axis=1)

    x = x[finite]
    y = y[finite]
    C = C[finite]

    if C.shape[1] > 0:
        keep = np.nanstd(C, axis=0) > 0
        C = C[:, keep]

    k = C.shape[1]
    n = len(x)
    df = n - k - 2
    if n < k + 4 or df <= 0 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, int(df), int(n)

    xr = residualize_vector(x, C) if C.shape[1] > 0 else x - np.mean(x)
    yr = residualize_vector(y, C) if C.shape[1] > 0 else y - np.mean(y)
    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        return np.nan, np.nan, int(df), int(n)

    r = float(np.corrcoef(xr, yr)[0, 1])
    if not np.isfinite(r):
        return np.nan, np.nan, int(df), int(n)

    r_clip = np.clip(r, -0.999999999, 0.999999999)
    t_stat = r_clip * np.sqrt(df / max(1e-12, 1.0 - r_clip ** 2))
    p_val = float(2.0 * student_t.sf(abs(t_stat), df))
    return r, p_val, int(df), int(n)


def get_partial_residual_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: List[str],
) -> Tuple[pd.DataFrame, float, float, int, int]:
    covariates = [c for c in covariates if c in df.columns]
    cols = ["subject", x_col, y_col] + covariates
    tmp = df[cols].dropna().copy()
    if tmp.empty:
        raise ValueError(
            f"No complete cases for {x_col}, {y_col}, covariates={covariates}"
        )

    x = tmp[x_col].astype(float).to_numpy()
    y = tmp[y_col].astype(float).to_numpy()
    C = (
        tmp[covariates].astype(float).to_numpy()
        if covariates
        else np.empty((len(tmp), 0))
    )

    if C.shape[1] > 0:
        keep = np.nanstd(C, axis=0) > 0
        C = C[:, keep]

    xr = residualize_vector(x, C) if C.shape[1] > 0 else x - np.mean(x)
    yr = residualize_vector(y, C) if C.shape[1] > 0 else y - np.mean(y)
    r, p, dfree, n = partial_corr_with_pvalue(x, y, C)

    out = tmp[["subject"]].copy()
    out[f"{x_col}_residual"] = xr
    out[f"{y_col}_residual"] = yr
    return out, r, p, dfree, n


def bootstrap_partial_corr_and_line_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covariates: List[str],
    grid: np.ndarray,
    n_boot: int = BOOTSTRAP_N_DEFAULT,
    seed: int = BOOTSTRAP_SEED_DEFAULT,
    ci: float = 95.0,
) -> Tuple[float, float, np.ndarray, np.ndarray, int]:
    """
    Participant-level percentile bootstrap for Panel C.

    For each bootstrap sample, participants are resampled with replacement,
    x and y are residualized against the covariates again, partial r is
    recomputed, and a residual-residual regression line is fitted. The returned
    band is the pointwise percentile interval across bootstrap fitted lines.
    """
    covariates = [c for c in covariates if c in df.columns]
    cols = [x_col, y_col] + covariates
    tmp = df[cols].copy()

    for col in cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

    x = tmp[x_col].to_numpy(dtype=float)
    y = tmp[y_col].to_numpy(dtype=float)
    C = (
        tmp[covariates].to_numpy(dtype=float)
        if covariates
        else np.empty((len(tmp), 0), dtype=float)
    )
    n = len(tmp)
    grid = np.asarray(grid, dtype=float)

    if n < 4 or n_boot <= 0:
        return (
            np.nan,
            np.nan,
            np.full_like(grid, np.nan, dtype=float),
            np.full_like(grid, np.nan, dtype=float),
            0,
        )

    feature_offset = sum(
        (i + 1) * ord(ch)
        for i, ch in enumerate(f"partial|{x_col}|{y_col}|{'|'.join(covariates)}")
    )
    rng = np.random.default_rng(
        (int(seed) + feature_offset) % (2**32 - 1)
    )

    boot_r = np.full(int(n_boot), np.nan, dtype=float)
    boot_lines = np.full((int(n_boot), len(grid)), np.nan, dtype=float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        Cb = C[idx, :] if C.ndim == 2 else np.empty((n, 0), dtype=float)

        if Cb.shape[1] > 0:
            keep = np.nanstd(Cb, axis=0) > 0
            Cb = Cb[:, keep]

        xr = residualize_vector(xb, Cb) if Cb.shape[1] > 0 else xb - np.mean(xb)
        yr = residualize_vector(yb, Cb) if Cb.shape[1] > 0 else yb - np.mean(yb)

        if (
            len(xr) < 4
            or np.nanstd(xr) <= 0
            or np.nanstd(yr) <= 0
        ):
            continue

        boot_r[b] = float(np.corrcoef(xr, yr)[0, 1])

        slope, intercept = np.polyfit(xr, yr, 1)
        boot_lines[b, :] = intercept + slope * grid

    valid_r = boot_r[np.isfinite(boot_r)]
    if valid_r.size == 0:
        r_lo = r_hi = np.nan
    else:
        alpha = (100.0 - float(ci)) / 2.0
        r_lo, r_hi = np.percentile(valid_r, [alpha, 100.0 - alpha])

    alpha = (100.0 - float(ci)) / 2.0
    valid_line_rows = np.any(np.isfinite(boot_lines), axis=1)
    valid_lines = boot_lines[valid_line_rows]

    if valid_lines.shape[0] == 0:
        line_lo = np.full_like(grid, np.nan, dtype=float)
        line_hi = np.full_like(grid, np.nan, dtype=float)
    else:
        with np.errstate(invalid="ignore"):
            line_lo = np.nanpercentile(valid_lines, alpha, axis=0)
            line_hi = np.nanpercentile(valid_lines, 100.0 - alpha, axis=0)

    return (
        float(r_lo) if np.isfinite(r_lo) else np.nan,
        float(r_hi) if np.isfinite(r_hi) else np.nan,
        line_lo,
        line_hi,
        int(valid_r.size),
    )


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.nanstd(x) == 0:
        return np.array([]), np.array([])
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ys = slope * xs + intercept
    return xs, ys


def _first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
    table_name: str,
    required: bool = True,
) -> Optional[str]:
    """Return the first available column name from a list of schema aliases."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"{table_name} does not contain any supported column among "
            f"{candidates}. Available columns: {list(df.columns)}"
        )
    return None


def _select_target_rows(
    df: pd.DataFrame,
    target: str,
    table_name: str,
) -> pd.DataFrame:
    """
    Select the requested target when the table contains a target column.

    New rebuilt null-distribution tables contain only one target and therefore
    may omit the target column entirely. In that case, the whole table is used.
    """
    if "target" not in df.columns:
        return df.copy()

    target_values = df["target"].astype(str).str.strip()
    requested = str(target).strip()

    out = df[target_values == requested].copy()
    if out.empty:
        # Be slightly forgiving about capitalization/whitespace.
        out = df[target_values.str.casefold() == requested.casefold()].copy()

    if out.empty:
        unique_targets = sorted(target_values.dropna().unique().tolist())
        raise ValueError(
            f"{table_name} contains no rows for target={target!r}. "
            f"Available targets: {unique_targets}"
        )
    return out


def draw_null_partial_distribution_panel(
    ax: plt.Axes,
    null_dist: pd.DataFrame,
    null_summary: pd.DataFrame,
    target: str,
    title: str,
    main_color: str,
    gray: str,
    line_color: str,
) -> Dict[str, float]:
    """
    Draw one partial-r null distribution.

    Supported schemas
    -----------------
    Original null output:
      distribution: target, null_partial_pearson_r
      summary: observed_partial_pearson_r, null_partial_r_2p5,
               null_partial_r_97p5, empirical_p_two_sided_abs

    Rebuilt QC null output:
      distribution: permutation, r, p, n, df, covariates
      summary: null_type, feature, target, observed_partial_r,
               empirical_p_two_sided_abs_r, ...

    For rebuilt outputs, the 2.5th and 97.5th percentiles are calculated
    directly from the permutation-level r values when they are not present in
    the summary table.
    """
    g = _select_target_rows(null_dist, target, f"{title} distribution")
    s = _select_target_rows(null_summary, target, f"{title} summary")

    if s.empty:
        raise ValueError(f"{title} summary contains no usable rows.")
    if len(s) > 1:
        # Prefer the primary mean-RR mNSRL row if multiple feature rows exist.
        if "feature" in s.columns:
            preferred = s[
                s["feature"].astype(str).isin(
                    ["observed_Lmax_mean_rrs", "negative_Lmax_mean_10_15_20"]
                )
            ]
            if not preferred.empty:
                s = preferred
        if len(s) > 1:
            print(
                f"[WARN] {title} summary has {len(s)} matching rows; "
                "the first row will be used."
            )

    dist_r_col = _first_existing_column(
        g,
        [
            "null_partial_pearson_r",  # original schema
            "r",                       # rebuilt QC schema
            "null_r",
            "partial_r",
        ],
        f"{title} distribution",
    )
    obs_r_col = _first_existing_column(
        s,
        [
            "observed_partial_pearson_r",  # original schema
            "observed_partial_r",          # rebuilt QC schema
            "observed_r",
        ],
        f"{title} summary",
    )
    emp_p_col = _first_existing_column(
        s,
        [
            "empirical_p_two_sided_abs",    # original schema
            "empirical_p_two_sided_abs_r",  # rebuilt QC schema
            "empirical_p_two_sided",
            "empirical_p",
        ],
        f"{title} summary",
    )

    vals_null = pd.to_numeric(g[dist_r_col], errors="coerce").dropna().to_numpy()
    if vals_null.size == 0:
        raise ValueError(
            f"{title} distribution column {dist_r_col!r} contains no finite values."
        )

    row = s.iloc[0]
    obs_r = float(row[obs_r_col])
    emp_p = float(row[emp_p_col])

    lo_col = _first_existing_column(
        s,
        ["null_partial_r_2p5", "null_r_2p5", "null_r_q025"],
        f"{title} summary",
        required=False,
    )
    hi_col = _first_existing_column(
        s,
        ["null_partial_r_97p5", "null_r_97p5", "null_r_q975"],
        f"{title} summary",
        required=False,
    )

    if lo_col is not None and hi_col is not None:
        lo = float(row[lo_col])
        hi = float(row[hi_col])
        interval_source = "summary"
    else:
        lo, hi = np.quantile(vals_null, [0.025, 0.975]).astype(float)
        interval_source = "permutation_distribution"

    null_mean_col = _first_existing_column(
        s,
        ["null_r_mean"],
        f"{title} summary",
        required=False,
    )
    null_sd_col = _first_existing_column(
        s,
        ["null_r_sd"],
        f"{title} summary",
        required=False,
    )
    null_mean = (
        float(row[null_mean_col])
        if null_mean_col is not None and np.isfinite(pd.to_numeric(row[null_mean_col], errors="coerce"))
        else float(np.mean(vals_null))
    )
    null_sd = (
        float(row[null_sd_col])
        if null_sd_col is not None and np.isfinite(pd.to_numeric(row[null_sd_col], errors="coerce"))
        else float(np.std(vals_null, ddof=1))
    )

    counts, bin_edges, patches = ax.hist(
        vals_null,
        bins=30,
        alpha=0.80,
        color=main_color,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(lo, linestyle="--", linewidth=1.1, color=gray, zorder=2)
    ax.axvline(hi, linestyle="--", linewidth=1.1, color=gray, zorder=2)
    ax.axvline(obs_r, linewidth=2.4, color=line_color, zorder=3)

    # Add headroom so the histogram does not overlap the upper-left statistics.
    max_count = float(np.max(counts)) if len(counts) else 1.0
    y_top = max_count * 1.32
    ax.set_ylim(0, y_top)

    # Place the label a fixed number of display points to the right of the line.
    # Using an offset in points is more stable than adding a small data-coordinate
    # increment and prevents the thick line from passing through the letters.
    ax.annotate(
        "Observed",
        xy=(obs_r, y_top * 0.67),
        xycoords="data",
        xytext=(8, 0),
        textcoords="offset points",
        rotation=90,
        rotation_mode="anchor",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="normal",
        color=line_color,
        zorder=4,
        clip_on=False,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.92),
    )
    ax.set_xlabel("Null partial " + r"$\it{r}$", fontsize=11, labelpad=5)
    ax.set_ylabel("Permutation count", fontsize=11, labelpad=5)
    ax.set_title(title, fontsize=12, fontweight="normal", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.05,
        0.95,
        rf"observed $\it{{r}}$ = {format_number(obs_r)}"
        + "\n"
        + rf"empirical $\it{{p}}$ = {format_number(emp_p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="normal",
    )

    schema_name = (
        "rebuilt_qc"
        if dist_r_col == "r" or obs_r_col == "observed_partial_r"
        else "original"
    )
    print(
        f"[INFO] {title}: schema={schema_name}, "
        f"distribution_r={dist_r_col}, observed_r={obs_r_col}, "
        f"empirical_p={emp_p_col}, CI_source={interval_source}, "
        f"n_perm={len(vals_null)}"
    )

    return {
        "observed_partial_r": obs_r,
        "null_r_mean": null_mean,
        "null_r_sd": null_sd,
        "null_r_2p5": float(lo),
        "null_r_97p5": float(hi),
        "empirical_p_two_sided_abs": emp_p,
        "n_permutations": int(len(vals_null)),
        "input_schema": schema_name,
        "distribution_r_column": dist_r_col,
        "observed_r_column": obs_r_col,
        "empirical_p_column": emp_p_col,
        "null_interval_source": interval_source,
    }


def create_main_figure(
    observed_subject_df: pd.DataFrame,
    shuffled_null_dist: pd.DataFrame,
    shuffled_null_summary: pd.DataFrame,
    negative_order_null_dist: pd.DataFrame,
    negative_order_null_summary: pd.DataFrame,
    target: str,
    feature: str,
    covariates: List[str],
    out_dir: Path,
    dpi: int = 300,
    include_negative_count: bool = False,
    sentence_null_title: str = "Sentence-shuffled null",
    label_null_title: str = "Label-shuffled null",
    bootstrap_n: int = BOOTSTRAP_N_DEFAULT,
    bootstrap_seed: int = BOOTSTRAP_SEED_DEFAULT,
) -> Dict[str, pd.DataFrame | Dict[str, float]]:
    """Create horizontal main-figure panels B-E."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11

    main_color = "#cd5c5c"
    gray = "#777777"
    line_color = "black"

    # Four-panel layout after removing the previous paired-null panel D.
    fig = plt.figure(figsize=(12.8, 3.25), dpi=dpi)

    # [left, bottom, width, height]
    axB = fig.add_axes([0.048, 0.265, 0.180, 0.600])
    axC = fig.add_axes([0.285, 0.245, 0.195, 0.620])
    axD = fig.add_axes([0.555, 0.265, 0.170, 0.600])
    axE = fig.add_axes([0.795, 0.265, 0.170, 0.600])

    # ------------------------------------------------------------------
    # Panel B: simple Pearson associations with ATQ
    # ------------------------------------------------------------------
    predictors = ["n_sentences", "negative_sentence_prop"]
    if include_negative_count and "negative_sentence_count" in observed_subject_df.columns:
        predictors.append("negative_sentence_count")
    predictors.append(feature)

    pearson_rows = []
    for pred in predictors:
        if pred not in observed_subject_df.columns or target not in observed_subject_df.columns:
            continue
        tmp = observed_subject_df[[pred, target]].dropna().copy()
        if tmp.shape[0] < 4 or tmp[pred].nunique() < 2 or tmp[target].nunique() < 2:
            r, p = np.nan, np.nan
        else:
            r, p = pearsonr(tmp[pred].astype(float), tmp[target].astype(float))

        ci_low, ci_high, bootstrap_valid = bootstrap_pearson_ci(
            observed_subject_df,
            pred,
            target,
            n_boot=bootstrap_n,
            seed=bootstrap_seed,
            ci=95.0,
        )

        pearson_rows.append(
            {
                "predictor": pred,
                "label": clean_label(pred),
                "axis_label": short_axis_label(pred),
                "n": len(tmp),
                "pearson_r": r,
                "pearson_p": p,
                "r2": r ** 2 if np.isfinite(r) else np.nan,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_n_requested": int(bootstrap_n),
                "bootstrap_n_valid": int(bootstrap_valid),
                "bootstrap_method": "participant percentile bootstrap 95% CI",
            }
        )
    pearson_df = pd.DataFrame(pearson_rows)

    xs = np.arange(len(pearson_df))
    vals = pearson_df["pearson_r"].astype(float).to_numpy()
    ci_low = pearson_df["bootstrap_ci_low"].astype(float).to_numpy()
    ci_high = pearson_df["bootstrap_ci_high"].astype(float).to_numpy()

    lower_err = np.where(
        np.isfinite(vals) & np.isfinite(ci_low),
        np.maximum(vals - ci_low, 0.0),
        0.0,
    )
    upper_err = np.where(
        np.isfinite(vals) & np.isfinite(ci_high),
        np.maximum(ci_high - vals, 0.0),
        0.0,
    )
    yerr = np.vstack([lower_err, upper_err])

    axB.axhline(0, linewidth=1.0, color="black")
    axB.bar(
        xs,
        vals,
        width=0.60,
        color=main_color,
        alpha=0.80,
        edgecolor="none",
        linewidth=0,
        yerr=yerr,
        capsize=3.0,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.0,
            "capthick": 1.0,
        },
    )
    axB.set_xticks(xs)
    axB.set_xticklabels(pearson_df["axis_label"], fontsize=9)
    axB.set_ylabel(r"$\it{r}$", fontsize=11, labelpad=5)
    axB.set_title("ATQ-30", fontsize=12, fontweight="normal", pad=8)
    axB.tick_params(axis="both", which="major", labelsize=10)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    finite_extent = np.concatenate([
        vals[np.isfinite(vals)],
        ci_low[np.isfinite(ci_low)],
        ci_high[np.isfinite(ci_high)],
    ])
    if finite_extent.size:
        data_min = float(np.nanmin(finite_extent))
        data_max = float(np.nanmax(finite_extent))
    else:
        data_min, data_max = -0.05, 0.50

    span = max(data_max - min(data_min, 0.0), 0.50)
    y_bottom = min(-0.05, data_min - 0.08 * span)
    y_top = max(0.50, data_max + 0.22 * span)
    axB.set_ylim(y_bottom, y_top)
    for x0, r0, p0, hi0 in zip(
        xs,
        pearson_df["pearson_r"],
        pearson_df["pearson_p"],
        pearson_df["bootstrap_ci_high"],
    ):
        if not np.isfinite(r0):
            continue
        st = fig1_significance_stars(p0)
        st = "" if st == "n.s." else st
        if st:
            top_value = float(hi0) if np.isfinite(hi0) else float(r0)
            axB.text(
                x0,
                top_value + 0.025,
                st,
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="normal",
            )

    # ------------------------------------------------------------------
    # Panel C: partial residual scatter
    # ------------------------------------------------------------------
    residual_df, r, p_part, dfree, n = get_partial_residual_dataframe(
        observed_subject_df,
        feature,
        target,
        covariates,
    )
    x_col = f"{feature}_residual"
    y_col = f"{target}_residual"
    x = residual_df[x_col].to_numpy(dtype=float)
    y = residual_df[y_col].to_numpy(dtype=float)

    axC.scatter(x, y, s=30, alpha=0.78, color=main_color, edgecolor="none")
    xs_fit, ys_fit = linear_fit(x, y)

    if len(xs_fit):
        ci_r_low, ci_r_high, line_ci_low, line_ci_high, bootstrap_valid_c = (
            bootstrap_partial_corr_and_line_ci(
                observed_subject_df,
                feature,
                target,
                covariates,
                grid=xs_fit,
                n_boot=bootstrap_n,
                seed=bootstrap_seed,
                ci=95.0,
            )
        )

        finite_band = np.isfinite(line_ci_low) & np.isfinite(line_ci_high)
        if finite_band.any():
            axC.fill_between(
                xs_fit[finite_band],
                line_ci_low[finite_band],
                line_ci_high[finite_band],
                color=main_color,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )

        axC.plot(xs_fit, ys_fit, linewidth=1.8, color=line_color, zorder=3)
    else:
        ci_r_low = np.nan
        ci_r_high = np.nan
        bootstrap_valid_c = 0

    axC.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.65)
    axC.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.65)
    axC.set_xlabel("mNSRL residual", fontsize=11, labelpad=5)
    axC.set_ylabel("ATQ-30 residual", fontsize=11, labelpad=5)
    axC.set_title("Adjusted association", fontsize=12, fontweight="normal", pad=8)
    axC.tick_params(axis="both", which="major", labelsize=10)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    def _fmt_stat_number(value: float, digits: int = 3) -> str:
        if not np.isfinite(value):
            return "NA"
        s = f"{value:.{digits}f}"
        if s.startswith("-0"):
            s = "-" + s[2:]
        elif s.startswith("0"):
            s = s[1:]
        return s

    stat_text = (
        "Partial " + r"$\it{r}$" + f" = {_fmt_stat_number(r, 3)}"
        + "\n"
        + "95% CI ["
        + _fmt_stat_number(ci_r_low, 3)
        + ", "
        + _fmt_stat_number(ci_r_high, 3)
        + "]"
        + "\n"
        + r"$\it{p}$"
        + f" = {_fmt_stat_number(p_part, 3)}"
    )
    axC.text(
        0.05,
        0.95,
        stat_text,
        transform=axC.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="normal",
    )
    residual_stats = {
        "partial_r": r,
        "partial_p": p_part,
        "partial_df": dfree,
        "n": n,
        "covariates": "+".join(covariates),
        "bootstrap_ci_low": ci_r_low,
        "bootstrap_ci_high": ci_r_high,
        "bootstrap_n_requested": int(bootstrap_n),
        "bootstrap_n_valid": int(bootstrap_valid_c),
        "bootstrap_method": (
            "participant percentile bootstrap 95% CI; "
            "covariate residualization and regression line re-estimated per resample"
        ),
    }

    # ------------------------------------------------------------------
    # Panel D: sentence-shuffled null distribution
    # ------------------------------------------------------------------
    shuffled_order_null_stats = draw_null_partial_distribution_panel(
        ax=axD,
        null_dist=shuffled_null_dist,
        null_summary=shuffled_null_summary,
        target=target,
        title=sentence_null_title,
        main_color=main_color,
        gray=gray,
        line_color=line_color,
    )

    # ------------------------------------------------------------------
    # Panel E: label-shuffled null distribution
    # ------------------------------------------------------------------
    negative_order_null_stats = draw_null_partial_distribution_panel(
        ax=axE,
        null_dist=negative_order_null_dist,
        null_summary=negative_order_null_summary,
        target=target,
        title=label_null_title,
        main_color=main_color,
        gray=gray,
        line_color=line_color,
    )

    panel_specs = [
        (axB, "B", -0.19, 1.18),
        (axC, "C", -0.17, 1.18),
        (axD, "D", -0.20, 1.18),
        (axE, "E", -0.20, 1.18),
    ]
    for ax, lab, xoff, yoff in panel_specs:
        ax.text(
            xoff,
            yoff,
            lab,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )

    png_path = out_dir / "Fig1.png"
    pdf_path = out_dir / "Fig1.pdf"
    fig.savefig(out_dir / "Fig1.tif", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    return {
        "pearson_df": pearson_df,
        "residual_df": residual_df,
        "residual_stats": residual_stats,
        "sentence_shuffled_null_stats": shuffled_order_null_stats,
        "label_shuffled_null_stats": negative_order_null_stats,
    }


# Figures 2 and 3 plotting utilities (original figure calculations preserved)

FONT_FAMILY = "Arial"
TEXT_SIZE = 12
PANEL_LABEL_SIZE = 18
BAR_WIDTH = 0.46
INDIANRED = "#cd5c5c"
BLUE = "#4f81bd"
GRAY = "#bdbdbd"
DARK_GRAY = "#555555"
GRID_GRAY = "#d9d9d9"
BLACK = "#000000"
WHITE = "#ffffff"
BASE_COVARIATES = ["n_sentences", "negative_sentence_prop", "age", "sex_M"]
PMC_ORDER = ["Precuneus", "PCC"]
MTL_ORDER = ["Anterior PHG", "Posterior PHG", "Hippocampus", "Amygdala"]
NODE_ORDER = PMC_ORDER + MTL_ORDER
SHORT_NODE = {"Precuneus": "Prec.", "PCC": "PCC", "Anterior PHG": "aPHG", "Posterior PHG": "pPHG", "Hippocampus": "Hipp.", "Amygdala": "Amyg."}
EDGE_ORDER = [f"{pmc}--{mtl}" for pmc in PMC_ORDER for mtl in MTL_ORDER]
EDGE_SHORT = {
    "Precuneus--Anterior PHG": "Prec.–aPHG", "Precuneus--Posterior PHG": "Prec.–pPHG",
    "Precuneus--Hippocampus": "Prec.–Hipp", "Precuneus--Amygdala": "Prec.–Amyg",
    "PCC--Anterior PHG": "PCC–aPHG", "PCC--Posterior PHG": "PCC–pPHG",
    "PCC--Hippocampus": "PCC–Hipp", "PCC--Amygdala": "PCC–Amyg",
}
PPHG_MODE1 = "node_contrib__mode1__Posterior_PHG"
plt.rcParams.update({"font.family": FONT_FAMILY, "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"], "font.size": TEXT_SIZE,
                     "axes.titlesize": TEXT_SIZE, "axes.labelsize": TEXT_SIZE, "xtick.labelsize": TEXT_SIZE,
                     "ytick.labelsize": TEXT_SIZE, "legend.fontsize": TEXT_SIZE, "figure.titlesize": TEXT_SIZE,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_panel_label(ax: plt.Axes, label: str, xoff: float = -0.14, yoff: float = 1.10) -> None:
    ax.text(
        xoff, yoff, label, transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE,
        fontweight="bold", ha="left", va="top", clip_on=False,
    )


def clean_axis(ax: plt.Axes, grid_y: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color=GRID_GRAY, linewidth=0.7, alpha=0.65)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, fig_dir: Path, stem: str, dpi: int) -> None:
    path = fig_dir / f"{stem}.png"
    fig.savefig(fig_dir / f"{stem}.tif", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    print("[FIGURE]", path)


def format_decimal(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    text = f"{float(value):.{digits}f}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return r"$p$ = NA"
    if value < .001:
        return r"$p$ < .001"
    return r"$p$ = " + format_decimal(value, 3)


def significance_stars(p: float) -> str:
    if not np.isfinite(p) or p >= .05:
        return ""
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    return "*"


def residualize(y: np.ndarray, C: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = C[:, None]
    if C.size == 0 or C.shape[1] == 0:
        return y - np.mean(y)
    X = np.column_stack([np.ones(len(y)), C])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def prepare_partial(frame: pd.DataFrame, x: str, y: str, covariates: Sequence[str]):
    cols = [x, y, *covariates]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    work = frame[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    used = [c for c in covariates if work[c].nunique(dropna=True) > 1]
    C = work[used].to_numpy(float) if used else np.empty((len(work), 0))
    return work, work[x].to_numpy(float), work[y].to_numpy(float), C


def partial_r(frame: pd.DataFrame, x: str, y: str, covariates: Sequence[str]) -> Tuple[float, float]:
    work, xv, yv, C = prepare_partial(frame, x, y, covariates)
    xr, yr = residualize(xv, C), residualize(yv, C)
    r = float(np.corrcoef(xr, yr)[0, 1])
    k = C.shape[1]
    df = len(work) - k - 2
    rc = np.clip(r, -.999999999, .999999999)
    t = rc * np.sqrt(df / (1 - rc * rc))
    p = float(2 * stats.t.sf(abs(t), df))
    return r, p


def bootstrap_scatter_band(
    frame: pd.DataFrame, x: str, y: str, covariates: Sequence[str],
    grid: np.ndarray, n_boot: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _, xv, yv, C = prepare_partial(frame, x, y, covariates)
    n = len(xv)
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        Cb = C[idx, :] if C.ndim == 2 else np.empty((n, 0))
        if Cb.size:
            keep = np.std(Cb, axis=0, ddof=1) > 0
            Cb = Cb[:, keep]
        xr = residualize(xv[idx], Cb)
        yr = residualize(yv[idx], Cb)
        if len(xr) < 2 or np.std(xr, ddof=1) <= 0:
            continue
        slope, intercept = np.polyfit(xr, yr, 1)
        curves.append(intercept + slope * grid)
    if not curves:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    return tuple(np.nanpercentile(np.asarray(curves), [2.5, 97.5], axis=0))


def draw_partial_scatter(
    ax: plt.Axes, frame: pd.DataFrame, x: str, y: str,
    covariates: Sequence[str], n_boot: int, seed: int,
    title: str, xlabel: str, ylabel: str,
    p_override: Optional[float] = None,
    ci_override: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    work, xv, yv, C = prepare_partial(frame, x, y, covariates)
    xr, yr = residualize(xv, C), residualize(yv, C)
    r, p = partial_r(frame, x, y, covariates)
    if p_override is not None:
        p = float(p_override)

    ax.scatter(xr, yr, s=36, alpha=.78, color=INDIANRED, edgecolor="none", zorder=3)
    if len(xr) >= 2 and np.std(xr, ddof=1) > 0:
        slope, intercept = np.polyfit(xr, yr, 1)
        grid = np.linspace(np.min(xr), np.max(xr), 100)
        blo, bhi = bootstrap_scatter_band(frame, x, y, covariates, grid, n_boot, seed)
        ax.fill_between(grid, blo, bhi, color=INDIANRED, alpha=.16, linewidth=0, zorder=1)
        ax.plot(grid, intercept + slope * grid, color=BLACK, lw=1.9, zorder=4)
    ax.axhline(0, color=GRAY, lw=.8, ls="--", alpha=.7)
    ax.axvline(0, color=GRAY, lw=.8, ls="--", alpha=.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ci_override is None:
        ci_lo = ci_hi = np.nan
    else:
        ci_lo, ci_hi = map(float, ci_override)
    ci_line = "" if not np.isfinite(ci_lo) else (
        f"\n95% CI [{format_decimal(ci_lo, 3)}, {format_decimal(ci_hi, 3)}]"
    )
    ax.text(
        .05, .95,
        f"Partial $r$ = {format_decimal(r, 3)}{ci_line}\n{format_p(p)}",
        transform=ax.transAxes, ha="left", va="top", fontsize=TEXT_SIZE,
    )
    clean_axis(ax, grid_y=False)
    return pd.DataFrame({
        "x_raw": xv, "y_raw": yv, "x_residual": xr, "y_residual": yr,
        "partial_r": r, "p_display": p, "ci95_low": ci_lo, "ci95_high": ci_hi,
        "bootstrap_seed": seed,
    })


def padded_ylim(values, lows, highs, markers) -> Tuple[float, float]:
    arrays = [np.asarray(values, float), np.asarray(lows, float), np.asarray(highs, float), np.array([0.0])]
    vals = np.concatenate([x[np.isfinite(x)] for x in arrays if np.isfinite(x).any()])
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, .25)
    return lo - .18 * span, hi + (.25 if any(markers) else .16) * span


def draw_assoc_bars(
    ax: plt.Axes, table: pd.DataFrame, labels: Sequence[str], title: str,
    ylabel: str, show_raw_only: bool = False, rotate: float = 0,
    raw_sig_marker: Optional[str] = None, use_fdr_markers: bool = True,
) -> None:
    vals = pd.to_numeric(table["partial_r"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(table["ci95_low"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(table["ci95_high"], errors="coerce").to_numpy(float)
    raw = pd.to_numeric(table["p_raw"], errors="coerce").to_numpy(float)
    q = pd.to_numeric(table.get("p_fdr", np.nan), errors="coerce").to_numpy(float)
    colors = [INDIANRED if v >= 0 else BLUE for v in vals]
    markers = []
    for pr, pq in zip(raw, q):
        mark = ""
        if raw_sig_marker is not None and np.isfinite(pr) and pr < .05:
            mark = raw_sig_marker
        elif use_fdr_markers and np.isfinite(pq):
            mark = significance_stars(pq)
        elif show_raw_only and np.isfinite(pr) and pr < .05:
            mark = "†"
        markers.append(mark)

    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, width=.62, alpha=.82, edgecolor="none", zorder=2)
    good = np.isfinite(vals) & np.isfinite(lo) & np.isfinite(hi)
    if good.any():
        ax.errorbar(
            x[good], vals[good],
            yerr=np.vstack([np.maximum(vals[good] - lo[good], 0), np.maximum(hi[good] - vals[good], 0)]),
            fmt="none", ecolor=BLACK, capsize=3, elinewidth=1.2, capthick=1.2, zorder=4,
        )
    ax.axhline(0, color=BLACK, lw=.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymin, ymax = padded_ylim(vals, lo, hi, [bool(m) for m in markers])
    ax.set_ylim(ymin, ymax)
    span = ymax - ymin
    for xi, v, lci, hci, m in zip(x, vals, lo, hi, markers):
        if not m:
            continue
        if v >= 0:
            y = min((hci if np.isfinite(hci) else v) + .035 * span, ymax - .055 * span)
            va = "bottom"
        else:
            y = max((lci if np.isfinite(lci) else v) - .035 * span, ymin + .055 * span)
            va = "top"
        ax.text(xi, y, m, ha="center", va=va, fontsize=TEXT_SIZE + 3, fontweight="bold")
    clean_axis(ax)


def participant_loading_summary(
    loadings: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize participant-specific node contributions from individual SVD fits.

    Regional contribution is defined exactly as in the primary localization
    analysis: squared singular-vector loading.  Squaring removes the arbitrary
    paired sign ambiguity of SVD and avoids imposing a group/common basis.
    Error bars are participant-bootstrap 95% confidence intervals for the mean.
    """
    required = {"subject", "mode", "partition", "node", "loading"}
    missing = required.difference(loadings.columns)
    if missing:
        raise ValueError(f"SVD loading table is missing columns: {sorted(missing)}")

    individual = loadings.copy()
    individual["mode"] = pd.to_numeric(individual["mode"], errors="coerce")
    individual["loading"] = pd.to_numeric(individual["loading"], errors="coerce")
    individual["squared_loading"] = individual["loading"] ** 2

    rows = []
    rng = np.random.default_rng(seed)
    for mode in [1, 2]:
        for node in NODE_ORDER:
            sub = individual[(individual["mode"] == mode) & (individual["node"] == node)].copy()
            vals = pd.to_numeric(sub["squared_loading"], errors="coerce").dropna().to_numpy(float)
            if len(vals) == 0:
                rows.append({
                    "mode": mode, "node": node, "n": 0,
                    "mean_squared_loading": np.nan,
                    "median_squared_loading": np.nan,
                    "sd_squared_loading": np.nan,
                    "ci95_low": np.nan, "ci95_high": np.nan,
                })
                continue

            if n_boot > 0 and len(vals) >= 2:
                boot_idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
                boot_mean = vals[boot_idx].mean(axis=1)
                ci_lo, ci_hi = np.percentile(boot_mean, [2.5, 97.5])
            else:
                ci_lo = ci_hi = np.nan

            rows.append({
                "mode": mode,
                "node": node,
                "n": len(vals),
                "mean_squared_loading": float(np.mean(vals)),
                "median_squared_loading": float(np.median(vals)),
                "sd_squared_loading": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                "ci95_low": float(ci_lo),
                "ci95_high": float(ci_hi),
            })

    return individual, pd.DataFrame(rows)


def draw_energy_panel(ax: plt.Axes, data: pd.DataFrame) -> pd.DataFrame:
    p = data[["subject", "svd_decomp__mode1_matrix_energy_share", "svd_decomp__mode2_matrix_energy_share"]].copy()
    p = p.rename(columns={"svd_decomp__mode1_matrix_energy_share": "Mode-1", "svd_decomp__mode2_matrix_energy_share": "Mode-2"}).dropna()
    x = np.array([0, 1])
    for _, row in p.iterrows():
        vals = row[["Mode-1", "Mode-2"]].astype(float).to_numpy()
        ax.plot(x, vals, color=GRAY, lw=.8, alpha=.34, zorder=1)
        ax.scatter(0, vals[0], s=24, color=INDIANRED, alpha=.40, edgecolor="none", zorder=2)
        ax.scatter(1, vals[1], s=24, color=BLUE, alpha=.40, edgecolor="none", zorder=2)
    means = p[["Mode-1", "Mode-2"]].mean().to_numpy(float)
    ax.scatter(x, means, s=72, marker="D", facecolors=[INDIANRED, BLUE], edgecolors=BLACK, linewidths=1, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(["Mode-1", "Mode-2"])
    ax.set_ylabel("Matrix energy fraction"); ax.set_title("Mode-specific matrix energy")
    ax.set_xlim(-.16, 1.16); ax.set_ylim(0, 1.05); clean_axis(ax)
    return pd.DataFrame({"mode": ["Mode-1", "Mode-2"], "mean": means,
                         "median": p[["Mode-1", "Mode-2"]].median().to_numpy(float),
                         "sd": p[["Mode-1", "Mode-2"]].std(ddof=1).to_numpy(float), "n": len(p)})


def draw_loading_profile(
    ax: plt.Axes,
    summary: pd.DataFrame,
    mode: int,
    ylim: Tuple[float, float],
) -> None:
    """Plot mean participant-specific regional contributions with bootstrap CIs.

    Regional contribution is the squared singular-vector loading used in the
    primary node-by-mode association analysis (Panel D). Panels B/C therefore
    summarize the across-participant distribution of the same quantity that is
    related to mNSRL in Panel D.
    """
    sub = summary[summary["mode"] == mode].set_index("node").reindex(NODE_ORDER).reset_index()
    vals = pd.to_numeric(sub["mean_squared_loading"], errors="coerce").to_numpy(float)
    lo = pd.to_numeric(sub["ci95_low"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(sub["ci95_high"], errors="coerce").to_numpy(float)
    x = np.arange(len(vals))
    mode_color = INDIANRED if mode == 1 else BLUE

    ax.bar(
        x, vals, color=mode_color, width=.66,
        alpha=.82, edgecolor="none", zorder=2,
    )
    good = np.isfinite(vals) & np.isfinite(lo) & np.isfinite(hi)
    if good.any():
        ax.errorbar(
            x[good], vals[good],
            yerr=np.vstack([
                np.maximum(vals[good] - lo[good], 0),
                np.maximum(hi[good] - vals[good], 0),
            ]),
            fmt="none", ecolor=BLACK, capsize=3,
            elinewidth=1.15, capthick=1.15, zorder=4,
        )

    ax.axhline(0, color=BLACK, lw=.9)
    # Separate PMC (first two nodes) from MTL (remaining four nodes).
    ax.axvline(1.5, color=BLACK, ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NODE[n] for n in NODE_ORDER], rotation=35, ha="right")
    ax.set_ylabel("Regional contribution")
    ax.set_title(f"Participant-specific Mode-{mode} contributions")
    ax.set_ylim(*ylim)
    clean_axis(ax, grid_y=True)
    ax.set_axisbelow(True)


def draw_grouped_node_mode(ax: plt.Axes, screen: pd.DataFrame) -> None:
    x = np.arange(len(NODE_ORDER), dtype=float)
    width = .36
    plotted = []
    all_v, all_lo, all_hi, all_m = [], [], [], []
    for mode, offset, color in [(1, -width/2, INDIANRED), (2, width/2, BLUE)]:
        s = screen[screen["mode"] == mode].set_index("node").reindex(NODE_ORDER).reset_index()
        v = pd.to_numeric(s["partial_r"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(s["ci95_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(s["ci95_high"], errors="coerce").to_numpy(float)
        q = pd.to_numeric(s["p_fdr"], errors="coerce").to_numpy(float)
        markers = [significance_stars(z) for z in q]
        xpos = x + offset
        ax.bar(xpos, v, width=width*.88, color=color, alpha=.82, edgecolor="none", label=f"Mode-{mode}", zorder=2)
        good = np.isfinite(v) & np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(xpos[good], v[good], yerr=np.vstack([v[good]-lo[good], hi[good]-v[good]]),
                    fmt="none", ecolor=BLACK, capsize=3, elinewidth=1.15, capthick=1.15, zorder=4)
        plotted.append((xpos, v, lo, hi, markers)); all_v += list(v); all_lo += list(lo); all_hi += list(hi); all_m += [bool(m) for m in markers]
    ax.axhline(0, color=BLACK, lw=.9); ax.axvline(1.5, color=BLACK, ls=":", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([SHORT_NODE[n] for n in NODE_ORDER])
    ax.set_ylabel("Partial " + r"$\it{r}$" + " with mNSRL")
    ax.set_title("Mode-specific regional contribution associations")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ymin, ymax = padded_ylim(all_v, all_lo, all_hi, all_m); ax.set_ylim(ymin, ymax); span = ymax-ymin
    for xpos, vals, lo, hi, marks in plotted:
        for xi, v, l, h, m in zip(xpos, vals, lo, hi, marks):
            if not m: continue
            y = min(h + .035*span, ymax-.055*span) if v >= 0 else max(l-.035*span, ymin+.055*span)
            ax.text(xi, y, m, ha="center", va="bottom" if v >= 0 else "top", fontsize=TEXT_SIZE+3, fontweight="bold")
    clean_axis(ax)


def create_figure3(data, loadings, screen, n_boot, seed, fig_dir, tab_dir, dpi, show):
    individual_loadings, loading_summary = participant_loading_summary(loadings, n_boot, seed)
    fig = plt.figure(figsize=(17.2, 10.0))
    ax_a = fig.add_axes([.06,.59,.20,.31]); ax_b = fig.add_axes([.34,.59,.25,.31]); ax_c = fig.add_axes([.69,.59,.25,.31])
    ax_d = fig.add_axes([.07,.10,.53,.32]); ax_e = fig.add_axes([.70,.10,.25,.32])
    energy_summary = draw_energy_panel(ax_a, data)

    limits = pd.to_numeric(
        loading_summary[["mean_squared_loading", "ci95_low", "ci95_high"]].stack(), errors="coerce"
    ).to_numpy(float)
    finite_limits = limits[np.isfinite(limits)]
    ml = max(.55, float(np.nanmax(finite_limits)) * 1.15) if len(finite_limits) else 1.0
    ylim = (0.0, min(1.0, ml))
    draw_loading_profile(ax_b, loading_summary, 1, ylim)
    draw_loading_profile(ax_c, loading_summary, 2, ylim)
    draw_grouped_node_mode(ax_d, screen)
    pphg = screen[(screen["mode"]==1)&(screen["node"]=="Posterior PHG")].iloc[0]
    resid = draw_partial_scatter(ax_e, data, PPHG_MODE1, TARGET, BASE_COVARIATES, n_boot, seed,
                                 "pPHG Mode-1 contribution vs. mNSRL",
                                 "pPHG Mode-1 contribution residual", "mNSRL residual",
                                 p_override=float(pphg["p_fdr"]),
                                 ci_override=(float(pphg["ci95_low"]), float(pphg["ci95_high"])))
    for l, ax, xo in [("A",ax_a,-.18),("B",ax_b,-.15),("C",ax_c,-.15),("D",ax_d,-.09),("E",ax_e,-.15)]:
        add_panel_label(ax,l,xo,1.11)
    save_figure(fig, fig_dir, "Fig3", dpi)
    if show: plt.show()
    plt.close(fig)
    energy_summary.to_csv(tab_dir/"Figure3A_mode_energy_summary.csv",index=False,encoding="utf-8-sig")
    loading_summary.to_csv(tab_dir/"Figure3BC_participant_specific_loading_summary.csv",index=False,encoding="utf-8-sig")
    individual_loadings.to_csv(tab_dir/"Figure3BC_participant_specific_loadings.csv",index=False,encoding="utf-8-sig")
    screen.to_csv(tab_dir/"Figure3D_node_by_mode_associations.csv",index=False,encoding="utf-8-sig")
    resid.assign(p_fdr=float(pphg["p_fdr"])).to_csv(tab_dir/"Figure3E_pphg_mode1_residuals.csv",index=False,encoding="utf-8-sig")
