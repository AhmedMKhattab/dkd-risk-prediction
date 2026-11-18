#!/usr/bin/env python3
"""
BioMe external validation for corrected DKD landmark models.

Frozen-analysis contract
------------------------
- BioMe is external validation only.
- No model fitting, hyperparameter tuning, recalibration, or BioMe-native
  threshold selection is performed here.
- Models, isotonic calibrators, exact feature order, and fixed-specificity
  thresholds are all learned in AoU and applied unchanged.
- Required predictors are never silently created. A missing required feature
  raises an error before scoring.

Primary external models
-----------------------
30d: M1, M2, M3, M4a
6m : M1, M2, M3, M4a
1y : M1, M2, M3, M4
2y : M1, M2, M3, M4

Secondary 1y models
-------------------
- Four-variable clinical logistic benchmark: age_at_t2d, eGFR_last,
  UACR_last, HbA1c_last
- M4 logistic regression
- M4 random forest

Collaborator run instructions
-----------------------------
1. The corrected R feature extraction has already been completed; do not rerun it.
2. Place JJ_biome_models_20260823.zip in /sc/arion/projects/loosr01a/JJ/dkd/.
   The script will verify and extract the exact 38 frozen AoU .pkl objects automatically.
3. Keep the AoU model and feature manifest CSVs in the same base directory.
4. Run this script. It performs preflight checks on all model objects and all four
   BioMe landmark files before scoring.
5. Return the single biome_external_return_bundle_20260823.zip created at the end.

Outputs contain dummy IDs only. Real BioMe person_id values are not written to
scoring outputs. Aggregate return-safe outputs additionally include manuscript Table 1
ingredients, feature/domain coverage, missingness by information layer and outcome,
observation-density summaries, and the exact BioMe data needed to recreate
Supplementary Figure 3 across all four landmarks.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import traceback
import platform
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb

from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy import stats as scipy_stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

RUN_DATE = "20260823"
ID_COL = "person_id"
Y_COL = "dkd_binary"

# -----------------------------------------------------------------------------
# FROZEN PYTHON ENVIRONMENT USED TO SERIALIZE THE AoU MODELS
# -----------------------------------------------------------------------------
# These are the package versions from the exact AoU environment that trained
# and pickled the external-validation models. sklearn/XGBoost pickle
# compatibility is not guaranteed across versions, so the validation script
# checks these before loading any model. Python patch-version differences are
# reported, while a different Python major/minor version stops the run.
EXPECTED_ENV = {
    "python_recommended": "3.10.19",
    "python_major_minor": "3.10",
    "numpy": "2.1.3",
    "pandas": "2.3.2",
    "scipy": "1.15.2",
    "scikit-learn": "1.6.0",
    "xgboost": "3.0.0",
}
# Pickle compatibility is most sensitive to the Python major/minor and the
# libraries that own the serialized estimators. We therefore stop on an exact
# scikit-learn or XGBoost mismatch, while reporting (but not blocking on)
# NumPy/pandas/SciPy differences. The accompanying requirements file records
# the complete AoU serialization environment.
STRICT_CORE_VERSION_CHECK = True
CORE_PICKLE_PACKAGES = ("scikit-learn", "xgboost")

def check_python_environment() -> dict:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
    }

    print("\n" + "=" * 78)
    print("PYTHON ENVIRONMENT CHECK")
    print("=" * 78)
    print(f"Python:        {actual['python']}  (recommended {EXPECTED_ENV['python_recommended']})")
    print(f"numpy:         {actual['numpy']}  (required {EXPECTED_ENV['numpy']})")
    print(f"pandas:        {actual['pandas']}  (required {EXPECTED_ENV['pandas']})")
    print(f"scipy:         {actual['scipy']}  (required {EXPECTED_ENV['scipy']})")
    print(f"scikit-learn:  {actual['scikit-learn']}  (required {EXPECTED_ENV['scikit-learn']})")
    print(f"xgboost:       {actual['xgboost']}  (required {EXPECTED_ENV['xgboost']})")

    py_mm = ".".join(actual["python"].split(".")[:2])
    if py_mm != EXPECTED_ENV["python_major_minor"]:
        raise RuntimeError(
            f"Python {actual['python']} detected, but these models were serialized under "
            f"Python {EXPECTED_ENV['python_recommended']}. Use Python "
            f"{EXPECTED_ENV['python_major_minor']}.x before scoring."
        )
    if actual["python"] != EXPECTED_ENV["python_recommended"]:
        print(
            f"WARNING: Python patch version differs from the training environment "
            f"({actual['python']} vs {EXPECTED_ENV['python_recommended']}). "
            "Package versions are checked exactly below."
        )

    package_keys = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost"]
    mismatches = [
        (k, actual[k], EXPECTED_ENV[k])
        for k in package_keys
        if actual[k] != EXPECTED_ENV[k]
    ]
    core_mismatches = [m for m in mismatches if m[0] in CORE_PICKLE_PACKAGES]
    soft_mismatches = [m for m in mismatches if m[0] not in CORE_PICKLE_PACKAGES]

    if soft_mismatches:
        msg = "; ".join(f"{k}: installed {a}, AoU {e}" for k, a, e in soft_mismatches)
        print("WARNING — non-core package version differs from AoU serialization environment:", msg)

    if core_mismatches:
        msg = "; ".join(f"{k}: installed {a}, required {e}" for k, a, e in core_mismatches)
        if STRICT_CORE_VERSION_CHECK:
            raise RuntimeError(
                "Core model-library version mismatch. These pickle files were created with "
                f"scikit-learn {EXPECTED_ENV['scikit-learn']} and XGBoost {EXPECTED_ENV['xgboost']}. "
                f"Please use those exact versions before scoring. {msg}"
            )
        print("WARNING — core model-library mismatch:", msg)

    print("Environment check passed.\n")
    return actual

# -----------------------------------------------------------------------------
# SECTION 0 — COLLABORATOR CONFIGURATION
# -----------------------------------------------------------------------------
# Put the R outputs, the two manifest CSVs, and the AoU model directory under
# BASE_DIR, or edit these paths once here.
BASE_DIR = Path(os.environ.get("BIOME_DKD_BASE_DIR", "/sc/arion/projects/loosr01a/JJ/dkd"))
OUT_DIR = BASE_DIR / f"biome_external_validation_{RUN_DATE}"

# JJ can simply place JJ_biome_models_20260823.zip in BASE_DIR. The script
# verifies its contents and extracts only the expected .pkl objects to
# BASE_DIR/models_v4. If the models are already extracted, several sensible
# locations are checked automatically.
MODEL_ZIP_NAME = f"JJ_biome_models_{RUN_DATE}.zip"
PREFERRED_MODEL_DIR = BASE_DIR / "models_v4"
MODEL_DIR_CANDIDATES = [
    PREFERRED_MODEL_DIR,
    BASE_DIR / "aou_models",
    BASE_DIR / f"JJ_biome_models_{RUN_DATE}",
    BASE_DIR,
]

FEATURE_MANIFEST_PATH = BASE_DIR / f"aou_biome_model_feature_manifest_{RUN_DATE}.csv"
MODEL_MANIFEST_PATH = BASE_DIR / f"aou_biome_model_manifest_{RUN_DATE}.csv"
THRESHOLD_TABLE_PATH = BASE_DIR / f"aou_biome_cal_thresholds_{RUN_DATE}.csv"
REQUIREMENTS_PATH = BASE_DIR / f"biome_python_requirements_{RUN_DATE}.txt"

BIOME_LANDMARK_PATHS = {
    "30d": BASE_DIR / "final_biome_landmark_30d.csv",
    "6m":  BASE_DIR / "final_biome_landmark_6m.csv",
    "1y":  BASE_DIR / "final_biome_landmark_1y.csv",
    "2y":  BASE_DIR / "final_biome_landmark_2y.csv",
}

# No bootstrap is required on the collaborator machine because the returned
# probability files contain everything needed for paired/bootstrap inference.
# Set True only if local confidence intervals are desired immediately.
RUN_BOOTSTRAP = False
N_BOOT = 2000
BOOT_SEED = 42

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def banner(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing model object: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def expected_model_files(model_manifest: pd.DataFrame) -> list[str]:
    files = []
    for col in ["raw_model_file", "calibrator_file"]:
        files.extend(model_manifest[col].astype(str).tolist())
    # Preserve order while enforcing uniqueness.
    return list(dict.fromkeys(files))


def _directory_has_all_models(directory: Path, expected: list[str]) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in expected)


def resolve_model_dir(model_manifest: pd.DataFrame) -> Path:
    expected = expected_model_files(model_manifest)
    if len(expected) != 38:
        raise RuntimeError(
            f"Model manifest should reference 38 pickle objects (19 raw + 19 calibrators); "
            f"found {len(expected)} unique filenames."
        )

    for directory in MODEL_DIR_CANDIDATES:
        if _directory_has_all_models(directory, expected):
            print(f"Found all {len(expected)} frozen AoU model objects in: {directory}")
            return directory

    zip_candidates = [
        BASE_DIR / MODEL_ZIP_NAME,
        Path.cwd() / MODEL_ZIP_NAME,
        Path(__file__).resolve().parent / MODEL_ZIP_NAME if "__file__" in globals() else BASE_DIR / MODEL_ZIP_NAME,
    ]
    zip_path = next((p for p in zip_candidates if p.is_file()), None)
    if zip_path is not None:
        print(f"Frozen AoU model objects not yet extracted. Found ZIP: {zip_path}")
        PREFERRED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        expected_set = set(expected)
        extracted = set()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                name = Path(member.filename).name
                if not name or name not in expected_set:
                    continue
                # Extract by basename only to avoid archive path surprises.
                target = PREFERRED_MODEL_DIR / name
                with zf.open(member, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.add(name)
        missing_from_zip = sorted(expected_set - extracted)
        if missing_from_zip:
            raise FileNotFoundError(
                f"{zip_path.name} is missing {len(missing_from_zip)} required model objects: "
                f"{missing_from_zip}"
            )
        print(f"Extracted and verified {len(extracted)} model objects to: {PREFERRED_MODEL_DIR}")
        return PREFERRED_MODEL_DIR

    missing_by_dir = {}
    for directory in MODEL_DIR_CANDIDATES:
        missing_by_dir[str(directory)] = [n for n in expected if not (directory / n).is_file()]
    first_missing = expected[:10]
    raise FileNotFoundError(
        f"Could not find the complete frozen AoU model package. Place {MODEL_ZIP_NAME} in "
        f"{BASE_DIR}, or extract all 38 .pkl files into {PREFERRED_MODEL_DIR}. "
        f"Example required files: {first_missing}"
    )


def write_model_inventory(model_manifest: pd.DataFrame, model_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for _, r in model_manifest.iterrows():
        for role, col in [("raw_model", "raw_model_file"), ("isotonic_calibrator", "calibrator_file")]:
            path = model_dir / str(r[col])
            rows.append({
                "analysis_group": str(r["analysis_group"]),
                "landmark": str(r["landmark"]),
                "model_id": str(r["model_id"]),
                "algorithm": str(r["algorithm"]),
                "object_role": role,
                "filename": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            })
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def extract_feature_names(estimator):
    """Best-effort extraction of fitted input feature names."""
    vals = getattr(estimator, "feature_names_in_", None)
    if vals is not None and len(vals):
        return [str(x) for x in vals]

    if hasattr(estimator, "get_booster"):
        try:
            vals = estimator.get_booster().feature_names
            if vals:
                return [str(x) for x in vals]
        except Exception:
            pass

    for attr in ("feature_names", "feature_name_", "feature_names_"):
        vals = getattr(estimator, attr, None)
        if vals is not None and len(vals):
            return [str(x) for x in vals]

    return None


def normalize_input_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return exact ordered numeric matrix without inventing missing columns."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"BioMe file is missing {len(missing)} REQUIRED model features. "
            f"First missing: {missing[:30]}"
        )

    X = df.loc[:, features].copy()

    # Historical BioMe files sometimes represented direction as text.
    slope_dir_cols = [c for c in features if c.endswith("_slope_dir")]
    for c in slope_dir_cols:
        X[c] = X[c].replace(
            {
                "up": 1, "UP": 1, "increase": 1, "increasing": 1,
                "flat": 0, "FLAT": 0, "stable": 0,
                "down": -1, "DOWN": -1, "decrease": -1, "decreasing": -1,
            }
        )

    # All frozen models were trained on numeric AoU matrices. Coercion to NaN
    # is allowed for malformed values; creation of an absent feature is not.
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X


def validate_model_features(model, features: list[str], label: str) -> None:
    fitted = extract_feature_names(model)
    if fitted is None:
        print(f"  [audit] {label}: fitted object exposes no feature-name metadata; manifest order used.")
        return
    if fitted != features:
        missing_in_manifest = [x for x in fitted if x not in features]
        extra_in_manifest = [x for x in features if x not in fitted]
        first_order_mismatch = next(
            ((i, a, b) for i, (a, b) in enumerate(zip(fitted, features)) if a != b),
            None,
        )
        raise ValueError(
            f"{label}: fitted feature names do not exactly match frozen manifest. "
            f"fitted_n={len(fitted)}, manifest_n={len(features)}, "
            f"first_order_mismatch={first_order_mismatch}, "
            f"missing_in_manifest={missing_in_manifest[:10]}, "
            f"extra_in_manifest={extra_in_manifest[:10]}"
        )


def safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return np.nan if np.unique(y).size < 2 else float(roc_auc_score(y, p))


def safe_auprc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return np.nan if np.unique(y).size < 2 else float(average_precision_score(y, p))


def oe_ratio(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    expected = float(np.sum(p))
    return np.nan if expected <= 0 else float(np.sum(y) / expected)


def calibration_intercept_slope(y, p):
    """Logistic recalibration intercept and slope; evaluation only."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = logit(p)
    X = np.column_stack([np.ones(len(z)), z])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(100):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        score = X.T @ (y - mu)
        hessian = -(X.T * w) @ X
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def calibration_in_the_large(y, p):
    """Calibration-in-the-large with slope fixed at 1."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = logit(p)
    target = float(np.sum(y))

    def score(alpha):
        return float(np.sum(expit(z + alpha)) - target)

    try:
        return float(brentq(score, -30.0, 30.0))
    except ValueError:
        return np.nan


def classification_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) else np.nan,
        "Accuracy": (tp + tn) / len(y) if len(y) else np.nan,
        "Positive_rate": (tp + fp) / len(y) if len(y) else np.nan,
    }


def stratified_bootstrap_indices(y, rng):
    y = np.asarray(y, dtype=int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return rng.choice(np.arange(len(y)), size=len(y), replace=True)
    idx = np.concatenate([
        rng.choice(idx0, size=len(idx0), replace=True),
        rng.choice(idx1, size=len(idx1), replace=True),
    ])
    rng.shuffle(idx)
    return idx


def pct_ci(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def bootstrap_metrics(y, p, threshold, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    store = {k: [] for k in [
        "AUROC", "AUPRC", "Brier", "O_E", "CITL",
        "Calibration_intercept", "Calibration_slope",
        "Sensitivity", "Specificity", "PPV", "NPV",
    ]}
    for _ in range(n_boot):
        idx = stratified_bootstrap_indices(y, rng)
        yb, pb = y[idx], p[idx]
        store["AUROC"].append(safe_auc(yb, pb))
        store["AUPRC"].append(safe_auprc(yb, pb))
        store["Brier"].append(float(brier_score_loss(yb, pb)))
        store["O_E"].append(oe_ratio(yb, pb))
        store["CITL"].append(calibration_in_the_large(yb, pb))
        a, b = calibration_intercept_slope(yb, pb)
        store["Calibration_intercept"].append(a)
        store["Calibration_slope"].append(b)
        cm = classification_metrics(yb, pb, threshold)
        for k in ["Sensitivity", "Specificity", "PPV", "NPV"]:
            store[k].append(cm[k])
    return {k: pct_ci(v) for k, v in store.items()}


# -----------------------------------------------------------------------------
# Feature-domain audit
# -----------------------------------------------------------------------------
COMORBIDITIES = {
    "acute_kidney_injury", "anxiety_disorder", "atrial_fibrillation",
    "chronic_hepatitis_c", "chronic_liver_disease", "cirrhosis", "copd",
    "coronary_artery_disease", "diabetic_foot", "diabetic_neuropathy",
    "diabetic_retinopathy", "gout", "heart_failure", "hiv", "hypertension",
    "major_depressive_disorder", "metabolic_syndrome", "myocardial_infarction",
    "nephrolithiasis", "obesity", "peripheral_arterial_disease", "psoriasis",
    "psychotic_disorder", "rheumatoid_arthritis", "sle", "sleep_apnea",
    "steatotic_liver_disease", "uti",
}

MED_PREFIXES = (
    "dpp4i_", "glp1ra_", "insulin_", "metformin_", "sglt2i_",
    "sulfonylureas_", "thiazolidinediones_", "htn_", "lipid_",
)

KIDNEY_PREFIXES = ("eGFR_", "Screatinine_", "UACR_", "UrAlb_")
VITAL_PREFIXES = ("SBP_", "DBP_", "HR_")
ANTHRO_PREFIXES = ("Weight_", "Height_")
GLYCEMIC_PREFIXES = ("HbA1c_", "RBG_")


def feature_domain(f: str) -> str:
    if f.startswith("PGS"):
        return "Genetics"
    if f in {"age_at_t2d", "sex_male"}:
        return "Demographics"
    if f in COMORBIDITIES:
        return "Comorbidities"
    if f.startswith(MED_PREFIXES):
        return "Treatment"
    if f.startswith(ANTHRO_PREFIXES):
        return "Anthropometrics"
    if f.startswith(VITAL_PREFIXES):
        return "Vitals"
    if f.startswith(KIDNEY_PREFIXES):
        return "Kidney measurements"
    if f.startswith(GLYCEMIC_PREFIXES):
        return "Glycemic measurements"
    if any(tok in f for tok in ("_recency_days", "_sd", "_cov", "_slope", "_slope_dir", "_ever_abnormal", "_frac_abnormal")):
        return "Other longitudinal measurements"
    return "Other laboratory measurements"


def feature_coverage_tables(df: pd.DataFrame, landmark: str, features: list[str]):
    rows = []
    for f in features:
        s = pd.to_numeric(df[f], errors="coerce")
        rows.append({
            "landmark": landmark,
            "feature": f,
            "domain": feature_domain(f),
            "n": int(len(s)),
            "n_nonmissing": int(s.notna().sum()),
            "nonmissing_fraction": float(s.notna().mean()),
            "n_zero_among_nonmissing": int((s.dropna() == 0).sum()),
            "zero_fraction_among_nonmissing": float((s.dropna() == 0).mean()) if s.notna().any() else np.nan,
        })
    per_feature = pd.DataFrame(rows)

    domain_rows = []
    for domain, feat_rows in per_feature.groupby("domain", sort=True):
        cols = feat_rows["feature"].tolist()
        block = df[cols].apply(pd.to_numeric, errors="coerce")
        participant_has_any = block.notna().any(axis=1)
        participant_all_missing = block.isna().all(axis=1)
        domain_rows.append({
            "landmark": landmark,
            "domain": domain,
            "n_features": len(cols),
            "n_participants": len(df),
            "cell_nonmissing_fraction": float(block.notna().to_numpy().mean()),
            "participants_with_any_observed_feature_fraction": float(participant_has_any.mean()),
            "participants_with_no_observed_feature_fraction": float(participant_all_missing.mean()),
            "mean_participant_nonmissing_fraction": float(block.notna().mean(axis=1).mean()),
        })
    return per_feature, pd.DataFrame(domain_rows)


# -----------------------------------------------------------------------------
# Table 1 / observability / collaborator-safe diagnostic summaries
# -----------------------------------------------------------------------------
# These outputs are intentionally aggregate only. They are designed so the
# collaborator can return one ZIP without sharing BioMe person identifiers.

TABLE1_SPEC = [
    # (feature, manuscript label, summary/test type)
    ("age_at_t2d",              "Age at first documented T2D, years", "gauss"),
    ("HbA1c_last",              "HbA1c, %",                           "gauss"),
    ("eGFR_last",               "eGFR, mL/min/1.73 m^2",              "gauss"),
    ("Screatinine_last",        "Creatinine, mg/dL",                  "gauss"),
    ("UACR_last",               "UACR, mg/g",                         "skewed"),
    ("UrAlb_last",              "Urine albumin",                      "skewed"),
    ("SBP_last",                "SBP, mmHg",                          "gauss"),
    ("DBP_last",                "DBP, mmHg",                          "gauss"),
    ("LDL_last",                "LDL, mg/dL",                         "gauss"),
    ("TG_last",                 "Triglycerides, mg/dL",               "skewed"),
    ("sex_male",                "Male sex, %",                        "binary"),
    ("hypertension",            "Hypertension, %",                    "binary"),
    ("coronary_artery_disease", "CAD, %",                             "binary"),
    ("heart_failure",           "Heart failure, %",                   "binary"),
    ("atrial_fibrillation",     "Atrial fibrillation, %",             "binary"),
    ("obesity",                 "Obesity, %",                         "binary"),
    ("acute_kidney_injury",     "AKI history, %",                     "binary"),
    ("htn_acei_ever",           "ACE inhibitor exposure, %",          "binary"),
    ("htn_arb_ever",            "ARB exposure, %",                    "binary"),
    ("sglt2i_ever",             "SGLT2 inhibitor exposure, %",        "binary"),
    ("glp1ra_ever",             "GLP-1RA exposure, %",                "binary"),
    ("insulin_ever",            "Insulin exposure, %",                "binary"),
    ("lipid_statins_ever",      "Statin exposure, %",                 "binary"),
]


def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "---"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_mean_sd(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "---"
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return f"{x.mean():.2f} ± {sd:.2f}" if np.isfinite(sd) else f"{x.mean():.2f}"


def _fmt_med_iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "---"
    return f"{x.median():.2f} ({x.quantile(.25):.2f}–{x.quantile(.75):.2f})"


def _fmt_pct(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "---"
    return f"{100.0 * (x == 1).mean():.1f}%"


def table1_long_summary(df: pd.DataFrame, landmark: str) -> pd.DataFrame:
    """Machine-readable Table 1 ingredients by case/control for one landmark."""
    rows = []
    for outcome_value, label in [(0, "Control"), (1, "Future DKD")]:
        sub = df[df[Y_COL] == outcome_value]
        for f, display, typ in TABLE1_SPEC:
            if f not in sub.columns:
                continue
            x = pd.to_numeric(sub[f], errors="coerce")
            valid = x.dropna()
            row = {
                "cohort": "BioMe", "landmark": landmark, "outcome": label,
                "feature": f, "characteristic": display, "variable_type": typ,
                "n_group": int(len(sub)), "n_nonmissing": int(valid.notna().sum()),
                "missing_fraction": float(x.isna().mean()),
                "mean": np.nan, "sd": np.nan, "median": np.nan,
                "q1": np.nan, "q3": np.nan,
                "n_positive": np.nan, "positive_fraction": np.nan,
            }
            if typ == "binary":
                row["n_positive"] = int((valid == 1).sum())
                row["positive_fraction"] = float((valid == 1).mean()) if len(valid) else np.nan
            else:
                row["mean"] = float(valid.mean()) if len(valid) else np.nan
                row["sd"] = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
                row["median"] = float(valid.median()) if len(valid) else np.nan
                row["q1"] = float(valid.quantile(.25)) if len(valid) else np.nan
                row["q3"] = float(valid.quantile(.75)) if len(valid) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def table1_formatted(df: pd.DataFrame, landmark: str) -> pd.DataFrame:
    """Manuscript-ready BioMe Table 1 columns with the same tests used previously."""
    d = df.copy()
    d[Y_COL] = pd.to_numeric(d[Y_COL], errors="raise").astype(int)
    ctrl = d[d[Y_COL] == 0]
    case = d[d[Y_COL] == 1]
    rows = []
    for f, display, typ in TABLE1_SPEC:
        if f not in d.columns:
            rows.append({
                "landmark": landmark, "Characteristic": display,
                "Controls": "---", "Cases": "---", "p-value": "---",
                "control_n_nonmissing": 0, "case_n_nonmissing": 0,
            })
            continue
        xc = pd.to_numeric(ctrl[f], errors="coerce")
        xs = pd.to_numeric(case[f], errors="coerce")
        vc, vs = xc.dropna(), xs.dropna()
        try:
            if typ == "gauss":
                p = scipy_stats.ttest_ind(vs, vc, equal_var=False, nan_policy="omit").pvalue
                cval, sval = _fmt_mean_sd(xc), _fmt_mean_sd(xs)
            elif typ == "skewed":
                p = scipy_stats.mannwhitneyu(vs, vc, alternative="two-sided").pvalue if len(vs) and len(vc) else np.nan
                cval, sval = _fmt_med_iqr(xc), _fmt_med_iqr(xs)
            else:
                cbin = vc.astype(int)
                sbin = vs.astype(int)
                tab = np.array([
                    [(cbin == 1).sum(), (cbin == 0).sum()],
                    [(sbin == 1).sum(), (sbin == 0).sum()],
                ])
                if tab.sum() == 0 or (tab.sum(axis=0) == 0).any() or (tab.sum(axis=1) == 0).any():
                    p = np.nan
                elif (tab < 5).any():
                    _, p = scipy_stats.fisher_exact(tab)
                else:
                    _, p, _, _ = scipy_stats.chi2_contingency(tab)
                cval, sval = _fmt_pct(xc), _fmt_pct(xs)
        except Exception:
            p = np.nan
            cval = _fmt_pct(xc) if typ == "binary" else (_fmt_med_iqr(xc) if typ == "skewed" else _fmt_mean_sd(xc))
            sval = _fmt_pct(xs) if typ == "binary" else (_fmt_med_iqr(xs) if typ == "skewed" else _fmt_mean_sd(xs))
        rows.append({
            "landmark": landmark, "Characteristic": display,
            "Controls": cval, "Cases": sval, "p-value": _fmt_p(float(p) if np.isfinite(p) else np.nan),
            "control_n_nonmissing": int(len(vc)), "case_n_nonmissing": int(len(vs)),
        })
    out = pd.DataFrame(rows)
    out.insert(1, "Controls_N", int((d[Y_COL] == 0).sum()))
    out.insert(2, "Cases_N", int((d[Y_COL] == 1).sum()))
    return out


# General, NONRENAL observation density used for observability summaries and
# Supplementary Figure 3. Kidney measures are intentionally excluded to avoid
# defining observability using the same biomarkers that contribute to the DKD endpoint.
OBS_DOMAIN_COUNTS = {
    "glycemic": ["HbA1c_n", "RBG_n"],
    "blood_pressure": ["SBP_n", "DBP_n"],
    "lipids": ["LDL_n", "HDL_n", "TG_n", "TotalChol_n"],
    "electrolytes": ["Na_n", "K_n", "Cl_n", "HCO3_n"],
    "liver_protein": ["ALT_n", "AST_n", "ALP_n", "Albumin_n", "TotalBilirubin_n", "DirectBilirubin_n"],
    "hematology": ["Hgb_n"],
    "anthropometrics": ["Weight_n"],
    "vitals": ["HR_n"],
}


def build_observation_density_work(df: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    work[Y_COL] = pd.to_numeric(df[Y_COL], errors="raise").astype(int)
    domain_cols = []
    count_cols = []
    for domain, cols in OBS_DOMAIN_COUNTS.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            work[f"count_{domain}"] = 0.0
            work[f"obs_{domain}"] = 0
        else:
            vals = df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            work[f"count_{domain}"] = vals.sum(axis=1)
            work[f"obs_{domain}"] = (vals.sum(axis=1) > 0).astype(int)
        count_cols.append(f"count_{domain}")
        domain_cols.append(f"obs_{domain}")
    work["general_measurement_count"] = work[count_cols].sum(axis=1)
    work["general_domains_observed"] = work[domain_cols].sum(axis=1)

    # Match dplyr::ntile-style behavior: always form three similarly sized
    # rank groups even when measurement counts are tied. Record count ranges
    # separately so the realized density represented by each group is auditable.
    ranks = work["general_measurement_count"].rank(method="first")
    work["observability_tertile"] = pd.qcut(ranks, q=3, labels=[1, 2, 3]).astype(int)
    return work


def observation_density_summary(df: pd.DataFrame, landmark: str) -> pd.DataFrame:
    work = build_observation_density_work(df)
    rows = []
    for out, label in [(0, "Control"), (1, "Future DKD")]:
        sub = work[work[Y_COL] == out]
        rows.append({
            "cohort": "BioMe", "landmark": landmark, "outcome": label, "n": len(sub),
            "mean_general_measurement_count": float(sub["general_measurement_count"].mean()),
            "median_general_measurement_count": float(sub["general_measurement_count"].median()),
            "q1_general_measurement_count": float(sub["general_measurement_count"].quantile(.25)),
            "q3_general_measurement_count": float(sub["general_measurement_count"].quantile(.75)),
            "mean_general_domains_observed": float(sub["general_domains_observed"].mean()),
            "median_general_domains_observed": float(sub["general_domains_observed"].median()),
        })
    return pd.DataFrame(rows)


def observability_tertile_summary(df: pd.DataFrame, landmark: str) -> pd.DataFrame:
    work = build_observation_density_work(df)
    rows = []
    for tert, sub in work.groupby("observability_tertile", sort=True):
        rows.append({
            "cohort": "BioMe", "landmark": landmark, "observability_tertile": int(tert),
            "n": int(len(sub)),
            "measurement_count_min": float(sub["general_measurement_count"].min()),
            "measurement_count_q1": float(sub["general_measurement_count"].quantile(.25)),
            "measurement_count_median": float(sub["general_measurement_count"].median()),
            "measurement_count_q3": float(sub["general_measurement_count"].quantile(.75)),
            "measurement_count_max": float(sub["general_measurement_count"].max()),
            "mean_domains_observed": float(sub["general_domains_observed"].mean()),
        })
    return pd.DataFrame(rows)


def coded_feature_prevalence_tables(df: pd.DataFrame, landmark: str):
    """Data behind Supplementary Figure 3 plus an unstratified prevalence audit."""
    work = build_observation_density_work(df)
    coded = []
    for f in sorted(COMORBIDITIES):
        if f in df.columns:
            coded.append((f, "coded_comorbidity"))
    med_ever = sorted([
        c for c in df.columns
        if c.endswith("_ever") and c.startswith(MED_PREFIXES)
    ])
    coded.extend((f, "medication_exposure") for f in med_ever)

    long_rows = []
    overall_rows = []
    for feature, feature_type in coded:
        vals = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        tmp = work[[Y_COL, "observability_tertile", "general_measurement_count", "general_domains_observed"]].copy()
        tmp[feature] = vals

        for out, outcome_label in [(0, "Control"), (1, "Future DKD")]:
            sub = tmp[tmp[Y_COL] == out]
            overall_rows.append({
                "cohort": "BioMe", "landmark": landmark,
                "feature": feature, "feature_type": feature_type,
                "outcome": outcome_label, "n": int(len(sub)),
                "n_positive": int((sub[feature] == 1).sum()),
                "prevalence_pct": float(100.0 * (sub[feature] == 1).mean()) if len(sub) else np.nan,
            })

        for (tert, out), sub in tmp.groupby(["observability_tertile", Y_COL], sort=True):
            long_rows.append({
                "cohort": "BioMe", "landmark": landmark,
                "feature": feature, "feature_type": feature_type,
                "observability_tertile": int(tert),
                "outcome": "Future DKD" if int(out) == 1 else "Control",
                "n": int(len(sub)),
                "n_positive": int((sub[feature] == 1).sum()),
                "prevalence_pct": float(100.0 * (sub[feature] == 1).mean()) if len(sub) else np.nan,
                "measurement_count_median": float(sub["general_measurement_count"].median()),
                "mean_domains_observed": float(sub["general_domains_observed"].mean()),
            })

    long_df = pd.DataFrame(long_rows)
    overall_df = pd.DataFrame(overall_rows)
    if long_df.empty:
        return overall_df, long_df, pd.DataFrame()

    gap = (
        long_df.pivot_table(
            index=["cohort", "landmark", "feature", "feature_type", "observability_tertile"],
            columns="outcome", values="prevalence_pct", aggfunc="first"
        )
        .reset_index()
    )
    if "Control" not in gap.columns:
        gap["Control"] = np.nan
    if "Future DKD" not in gap.columns:
        gap["Future DKD"] = np.nan
    gap = gap.rename(columns={"Control": "control_pct", "Future DKD": "case_pct"})
    gap["case_minus_control_pct"] = gap["case_pct"] - gap["control_pct"]
    gap["abs_case_control_gap_pct"] = gap["case_minus_control_pct"].abs()
    gap = gap.sort_values(["landmark", "feature_type", "feature", "observability_tertile"])
    return overall_df, long_df, gap


def feature_coverage_by_outcome(df: pd.DataFrame, landmark: str, features: list[str]):
    feature_rows = []
    domain_rows = []
    for out, label in [(0, "Control"), (1, "Future DKD")]:
        sub = df[df[Y_COL] == out]
        for f in features:
            s = pd.to_numeric(sub[f], errors="coerce")
            feature_rows.append({
                "cohort": "BioMe", "landmark": landmark, "outcome": label,
                "feature": f, "domain": feature_domain(f), "n": int(len(s)),
                "n_nonmissing": int(s.notna().sum()),
                "nonmissing_fraction": float(s.notna().mean()),
                "n_unique_nonmissing": int(s.dropna().nunique()),
                "zero_fraction_among_nonmissing": float((s.dropna() == 0).mean()) if s.notna().any() else np.nan,
            })
        fdf = pd.DataFrame(feature_rows)
        # Only summarize rows for this outcome to avoid reusing prior group rows.
        fsub = fdf[(fdf["landmark"] == landmark) & (fdf["outcome"] == label)]
        for domain, g in fsub.groupby("domain", sort=True):
            cols = g["feature"].tolist()
            block = sub[cols].apply(pd.to_numeric, errors="coerce")
            domain_rows.append({
                "cohort": "BioMe", "landmark": landmark, "outcome": label,
                "domain": domain, "n_features": len(cols), "n_participants": len(sub),
                "cell_nonmissing_fraction": float(block.notna().to_numpy().mean()),
                "mean_participant_nonmissing_fraction": float(block.notna().mean(axis=1).mean()),
                "participants_with_any_observed_feature_fraction": float(block.notna().any(axis=1).mean()),
                "participants_with_no_observed_feature_fraction": float(block.isna().all(axis=1).mean()),
            })
    return pd.DataFrame(feature_rows), pd.DataFrame(domain_rows)


def incremental_layer_missingness(df: pd.DataFrame, landmark: str, feature_manifest: pd.DataFrame) -> pd.DataFrame:
    """Missingness in the incremental information added at each nested primary model."""
    primary = feature_manifest[
        (feature_manifest["analysis_group"] == "primary_biome_compatible")
        & (feature_manifest["landmark"] == landmark)
    ].copy()
    sequence = ["M1", "M2", "M3", "M4a" if landmark in {"30d", "6m"} else "M4"]
    previous = set()
    rows = []
    for model_id in sequence:
        fm = primary[primary["model_id"] == model_id].sort_values("feature_order")
        current = fm["feature"].astype(str).tolist()
        added = [f for f in current if f not in previous]
        previous = set(current)
        if not added:
            continue
        for out, label in [(0, "Control"), (1, "Future DKD")]:
            sub = df[df[Y_COL] == out]
            block = sub[added].apply(pd.to_numeric, errors="coerce")
            rows.append({
                "cohort": "BioMe", "landmark": landmark, "model_id": model_id,
                "incremental_layer": model_id, "outcome": label,
                "n_participants": int(len(sub)), "n_features_added": int(len(added)),
                "cell_missing_fraction": float(block.isna().to_numpy().mean()),
                "mean_participant_missing_fraction": float(block.isna().mean(axis=1).mean()),
                "participants_with_no_observed_features_in_layer_fraction": float(block.isna().all(axis=1).mean()),
                "participants_with_any_observed_feature_in_layer_fraction": float(block.notna().any(axis=1).mean()),
            })
    return pd.DataFrame(rows)

def normalize_and_validate_coded_binary_features(
    df: pd.DataFrame, landmark: str
) -> list[dict]:
    """Normalize intended 0/1 EHR flags and hard-check malformed values.

    BioMe medication indicators are defined as *recorded exposure* flags. In the
    R extractor, participants with at least one medication-source row are
    explicitly given 0 for drug classes never observed. Participants with no
    medication-source row at all can acquire NA only at the final left join.
    Those structural NAs therefore have the same intended meaning, no recorded
    exposure, and are deterministically coded to 0 here.

    This function does NOT coerce malformed nonmissing text to 0. Any nonnumeric
    nonmissing value or any value outside {0,1} remains a hard failure. Sex and
    coded comorbidities are also kept strict because their R extractors already
    construct complete cohort-wide fields.
    """
    audit = []

    med_candidates = [
        c for c in df.columns
        if c.startswith(MED_PREFIXES)
        and (c.endswith("_ever") or c.endswith("_recent_30d"))
    ]

    # Medication flags: structural missingness from the final cohort left join
    # means no recorded exposure and should be 0, matching the R feature
    # definition for participants represented in the medication table.
    for c in med_candidates:
        raw = df[c]
        numeric = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & numeric.isna()
        if malformed.any():
            examples = raw[malformed].astype(str).head(5).tolist()
            raise ValueError(
                f"{landmark}: medication feature {c} contains "
                f"{int(malformed.sum())} nonnumeric nonmissing value(s); "
                f"examples={examples}"
            )
        bad = sorted(set(numeric.dropna().unique()) - {0, 1, 0.0, 1.0})
        if bad:
            raise ValueError(
                f"{landmark}: medication feature {c} has unexpected values {bad[:10]}"
            )
        n_missing = int(numeric.isna().sum())
        if n_missing:
            numeric = numeric.fillna(0.0)
            print(
                f"  [coded-feature normalization] {c}: "
                f"filled {n_missing} structural NA(s) with 0 (no recorded exposure)"
            )
        df[c] = numeric.astype(int)
        audit.append({
            "landmark": landmark,
            "feature": c,
            "feature_type": "medication_exposure_flag",
            "n_structural_missing_filled_zero": n_missing,
        })

    # Sex and comorbidity flags should already be complete after 07B. Do not
    # invent values for these fields.
    strict_candidates = ["sex_male"]
    strict_candidates += [c for c in sorted(COMORBIDITIES) if c in df.columns]
    problems = []
    for c in strict_candidates:
        raw = df[c]
        numeric = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & numeric.isna()
        if malformed.any():
            problems.append(f"{c}: {int(malformed.sum())} nonnumeric")
            continue
        if numeric.isna().any():
            problems.append(f"{c}: {int(numeric.isna().sum())} missing")
            continue
        bad = sorted(set(numeric.unique()) - {0, 1, 0.0, 1.0})
        if bad:
            problems.append(f"{c}: unexpected values {bad[:10]}")
        else:
            df[c] = numeric.astype(int)

    if problems:
        raise ValueError(
            f"{landmark}: strict coded binary feature QC failed. "
            + "; ".join(problems[:20])
        )

    return audit


def make_dummy_ids(n: int, landmark: str) -> list[str]:
    """Deterministic pseudonyms unrelated to the real person_id ordering."""
    lm_seed = {"30d": 30042, "6m": 60042, "1y": 100042, "2y": 200042}[landmark]
    rng = np.random.default_rng(lm_seed)
    labels = np.arange(1, n + 1)
    rng.shuffle(labels)
    return [f"BIOME_{landmark}_{i:06d}" for i in labels]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    environment_actual = check_python_environment()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    env_path = OUT_DIR / f"biome_python_environment_{RUN_DATE}.txt"
    env_lines = [
        "BioMe external-validation Python environment",
        f"Python={environment_actual['python']}",
        f"numpy={environment_actual['numpy']}",
        f"pandas={environment_actual['pandas']}",
        f"scipy={environment_actual['scipy']}",
        f"scikit-learn={environment_actual['scikit-learn']}",
        f"xgboost={environment_actual['xgboost']}",
        "",
        "Expected model-serialization environment:",
        f"Python={EXPECTED_ENV['python_recommended']}",
        f"numpy={EXPECTED_ENV['numpy']}",
        f"pandas={EXPECTED_ENV['pandas']}",
        f"scipy={EXPECTED_ENV['scipy']}",
        f"scikit-learn={EXPECTED_ENV['scikit-learn']}",
        f"xgboost={EXPECTED_ENV['xgboost']}",
    ]
    env_path.write_text("\n".join(env_lines) + "\n")
    banner("BioMe DKD external validation — frozen AoU models")

    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OUT_DIR: {OUT_DIR}")

    for path in [FEATURE_MANIFEST_PATH, MODEL_MANIFEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required manifest missing: {path}")

    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    model_manifest = pd.read_csv(MODEL_MANIFEST_PATH)

    required_fm_cols = {
        "analysis_group", "landmark", "model_id", "algorithm",
        "feature_order", "feature",
    }
    required_mm_cols = {
        "analysis_group", "landmark", "model_id", "algorithm", "n_features",
        "raw_model_file", "calibrator_file",
        "threshold_spec_80", "threshold_spec_85", "threshold_spec_90", "threshold_spec_95",
    }
    if not required_fm_cols.issubset(feature_manifest.columns):
        raise ValueError(f"Feature manifest missing columns: {sorted(required_fm_cols - set(feature_manifest.columns))}")
    if not required_mm_cols.issubset(model_manifest.columns):
        raise ValueError(f"Model manifest missing columns: {sorted(required_mm_cols - set(model_manifest.columns))}")

    # Exact manifest row-count check.
    counts = (
        feature_manifest
        .groupby(["analysis_group", "landmark", "model_id", "algorithm"])
        .size()
        .reset_index(name="manifest_n_features")
    )
    checked = model_manifest.merge(
        counts,
        on=["analysis_group", "landmark", "model_id", "algorithm"],
        how="left",
        validate="one_to_one",
    )
    if checked["manifest_n_features"].isna().any() or not (
        checked["n_features"].astype(int) == checked["manifest_n_features"].astype(int)
    ).all():
        raise ValueError("Model/feature manifest feature-count mismatch")

    # If the standalone AoU threshold table is present, verify it agrees with
    # the model manifest so a stale support CSV cannot silently change policy.
    if THRESHOLD_TABLE_PATH.exists():
        threshold_table = pd.read_csv(THRESHOLD_TABLE_PATH)
        required_thr_cols = {
            "analysis_group", "landmark", "model_id", "algorithm",
            "target_specificity", "threshold_cal",
        }
        if not required_thr_cols.issubset(threshold_table.columns):
            raise ValueError(
                f"Threshold table missing columns: "
                f"{sorted(required_thr_cols - set(threshold_table.columns))}"
            )
        keys = ["analysis_group", "landmark", "model_id", "algorithm"]
        for _, mr in model_manifest.iterrows():
            sub = threshold_table.copy()
            for k in keys:
                sub = sub[sub[k].astype(str) == str(mr[k])]
            if len(sub) != 4:
                raise ValueError(
                    f"Threshold table should contain 4 policy rows for "
                    f"{mr['analysis_group']}/{mr['landmark']}/{mr['model_id']}/{mr['algorithm']}; "
                    f"found {len(sub)}"
                )
            for pct in (80, 85, 90, 95):
                target = pct / 100.0
                rr = sub[np.isclose(pd.to_numeric(sub["target_specificity"], errors="coerce"), target)]
                if len(rr) != 1:
                    raise ValueError(
                        f"Threshold table missing/duplicating target specificity {target:.2f} for "
                        f"{mr['analysis_group']}/{mr['landmark']}/{mr['model_id']}/{mr['algorithm']}"
                    )
                a = float(mr[f"threshold_spec_{pct}"])
                b = float(rr.iloc[0]["threshold_cal"])
                if not np.isclose(a, b, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"Threshold mismatch for {mr['analysis_group']}/{mr['landmark']}/"
                        f"{mr['model_id']}/{mr['algorithm']} at {pct}% specificity: "
                        f"model_manifest={a}, threshold_table={b}"
                    )
        print("Threshold-table cross-check passed for all frozen models and policies.")

    # Resolve the 38 frozen AoU objects. If JJ places the single model ZIP in
    # BASE_DIR, it is extracted automatically and verified against the manifest.
    model_dir = resolve_model_dir(model_manifest)
    print(f"MODEL_DIR: {model_dir}")

    model_inventory_path = OUT_DIR / f"biome_external_model_inventory_{RUN_DATE}.csv"
    model_inventory_df = write_model_inventory(model_manifest, model_dir, model_inventory_path)
    print(f"Verified {len(model_inventory_df)} model/calibrator objects and wrote SHA256 inventory.")

    # Load and validate every model/calibrator before touching the BioMe scoring
    # loop. This catches a corrupted/incompatible object up front rather than
    # after several landmarks have already been processed.
    model_cache = {}
    banner("MODEL PACKAGE PREFLIGHT")
    for _, mr in model_manifest.iterrows():
        group = str(mr["analysis_group"])
        lm = str(mr["landmark"])
        model_id = str(mr["model_id"])
        alg = str(mr["algorithm"])
        key = (group, lm, model_id, alg)
        label = f"{group}/{lm}/{model_id}/{alg}"
        fm = feature_manifest[
            (feature_manifest["analysis_group"] == group)
            & (feature_manifest["landmark"].astype(str) == lm)
            & (feature_manifest["model_id"].astype(str) == model_id)
            & (feature_manifest["algorithm"].astype(str) == alg)
        ].sort_values("feature_order")
        features = fm["feature"].astype(str).tolist()
        raw_model = load_pickle(model_dir / str(mr["raw_model_file"]))
        calibrator = load_pickle(model_dir / str(mr["calibrator_file"]))
        validate_model_features(raw_model, features, label + "/raw")
        validate_model_features(calibrator, features, label + "/calibrator")
        model_cache[key] = (raw_model, calibrator)
        print(f"  OK  {label} | {len(features)} features")
    print(f"All {len(model_cache)} frozen models and calibrators loaded successfully.\n")

    # Input existence before any model scoring.
    for lm, path in BIOME_LANDMARK_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing BioMe landmark CSV for {lm}: {path}")

    # Model object existence has already been verified by resolve_model_dir().

    perf_rows = []
    policy_rows = []
    raw_cal_rows = []
    all_feature_rows = []
    all_domain_rows = []
    all_feature_outcome_rows = []
    all_domain_outcome_rows = []
    all_layer_missingness_rows = []
    reclass_rows = []
    cohort_rows = []
    table1_rows = []
    table1_formatted_rows = []
    obs_rows = []
    obs_tertile_rows = []
    coded_overall_rows = []
    suppfig3_long_rows = []
    suppfig3_gap_rows = []
    scored_by_landmark = {}
    binary_normalization_rows = []

    for lm, csv_path in BIOME_LANDMARK_PATHS.items():
        banner(f"LANDMARK {lm}")
        df = pd.read_csv(csv_path, low_memory=False)
        if ID_COL not in df.columns or Y_COL not in df.columns:
            raise ValueError(f"{lm}: input must contain {ID_COL} and {Y_COL}")
        if df[ID_COL].isna().any() or df[ID_COL].duplicated().any():
            raise ValueError(f"{lm}: person_id must be nonmissing and unique")

        y = pd.to_numeric(df[Y_COL], errors="raise").astype(int)
        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"{lm}: {Y_COL} must be binary 0/1")
        df[Y_COL] = y
        binary_normalization_rows.extend(
            normalize_and_validate_coded_binary_features(df, lm)
        )

        cohort_rows.append({
            "landmark": lm,
            "n": len(df),
            "cases": int(y.sum()),
            "controls": int((y == 0).sum()),
            "prevalence": float(y.mean()),
        })
        print(f"N={len(df):,} | cases={int(y.sum()):,} | controls={int((y==0).sum()):,} | prevalence={y.mean():.3f}")

        # Domain coverage is evaluated on the union of frozen primary predictors
        # for that landmark, not on arbitrary extra BioMe columns.
        primary_fm = feature_manifest[
            (feature_manifest["analysis_group"] == "primary_biome_compatible")
            & (feature_manifest["landmark"] == lm)
        ]
        primary_union = list(dict.fromkeys(primary_fm.sort_values("feature_order")["feature"].tolist()))
        missing_union = [c for c in primary_union if c not in df.columns]
        if missing_union:
            raise ValueError(f"{lm}: missing frozen primary feature(s): {missing_union[:30]}")
        pf, pdomain = feature_coverage_tables(df, lm, primary_union)
        all_feature_rows.append(pf)
        all_domain_rows.append(pdomain)

        # Aggregate summaries needed for manuscript tables, observability analyses,
        # reviewer responses, and Supplementary Figure 3. Generate them at ALL
        # four landmarks so no collaborator rerun is needed later.
        table1_rows.append(table1_long_summary(df, lm))
        table1_formatted_rows.append(table1_formatted(df, lm))
        obs_rows.append(observation_density_summary(df, lm))
        obs_tertile_rows.append(observability_tertile_summary(df, lm))

        pf_out, pdomain_out = feature_coverage_by_outcome(df, lm, primary_union)
        all_feature_outcome_rows.append(pf_out)
        all_domain_outcome_rows.append(pdomain_out)
        all_layer_missingness_rows.append(incremental_layer_missingness(df, lm, feature_manifest))

        coded_overall, supp_long, supp_gap = coded_feature_prevalence_tables(df, lm)
        coded_overall_rows.append(coded_overall)
        suppfig3_long_rows.append(supp_long)
        suppfig3_gap_rows.append(supp_gap)

        # Dummy IDs: deterministic within the landmark but unrelated to the
        # ordering or numeric value of the real BioMe identifier.
        dummy_ids = make_dummy_ids(len(df), lm)
        prob_out = pd.DataFrame({"dummy_id": dummy_ids, Y_COL: y.to_numpy()})

        lm_models = model_manifest[model_manifest["landmark"] == lm].copy()
        # Primary first, then secondary.
        group_order = {"primary_biome_compatible": 0, "clinical_benchmark": 1, "algorithmic_sensitivity": 2}
        lm_models["_ord"] = lm_models["analysis_group"].map(group_order).fillna(9)
        lm_models = lm_models.sort_values(["_ord", "model_id", "algorithm"])

        class_at_90 = {}

        for _, mr in lm_models.iterrows():
            group = str(mr["analysis_group"])
            model_id = str(mr["model_id"])
            alg = str(mr["algorithm"])
            key = (group, lm, model_id, alg)
            label = f"{group}/{lm}/{model_id}/{alg}"

            fm = feature_manifest[
                (feature_manifest["analysis_group"] == group)
                & (feature_manifest["landmark"] == lm)
                & (feature_manifest["model_id"] == model_id)
                & (feature_manifest["algorithm"] == alg)
            ].sort_values("feature_order")
            features = fm["feature"].astype(str).tolist()
            if len(features) != int(mr["n_features"]):
                raise ValueError(f"{label}: manifest count mismatch")
            if len(features) != len(set(features)):
                raise ValueError(f"{label}: duplicate feature names in manifest")

            X = normalize_input_values(df, features)
            raw_path = model_dir / str(mr["raw_model_file"])
            cal_path = model_dir / str(mr["calibrator_file"])

            if key not in model_cache:
                raise RuntimeError(f"{label}: model missing from completed preflight cache")
            raw_model, calibrator = model_cache[key]

            p_raw = np.asarray(raw_model.predict_proba(X)[:, 1], dtype=float)
            p_cal = np.asarray(calibrator.predict_proba(X)[:, 1], dtype=float)
            if not (np.isfinite(p_raw).all() and np.isfinite(p_cal).all()):
                raise ValueError(f"{label}: non-finite predicted probability")

            col_stem = f"{group}__{model_id}__{alg}"
            prob_out[f"{col_stem}__p_raw"] = p_raw
            prob_out[f"{col_stem}__p_calibrated"] = p_cal

            raw_cal_rows.append({
                "analysis_group": group, "landmark": lm, "model_id": model_id, "algorithm": alg,
                "n_features": len(features), "n": len(y), "cases": int(y.sum()),
                "raw_AUROC": safe_auc(y, p_raw), "calibrated_AUROC": safe_auc(y, p_cal),
                "raw_AUPRC": safe_auprc(y, p_raw), "calibrated_AUPRC": safe_auprc(y, p_cal),
                "raw_Brier": float(brier_score_loss(y, p_raw)),
                "calibrated_Brier": float(brier_score_loss(y, p_cal)),
            })

            cal_intercept, cal_slope = calibration_intercept_slope(y, p_cal)
            threshold_90 = float(mr["threshold_spec_90"])
            cm90 = classification_metrics(y, p_cal, threshold_90)

            row = {
                "analysis_group": group, "landmark": lm, "model_id": model_id, "algorithm": alg,
                "n_features": len(features), "n": len(y), "cases": int(y.sum()), "controls": int((y == 0).sum()),
                "prevalence": float(y.mean()),
                "AUROC": safe_auc(y, p_cal), "AUPRC": safe_auprc(y, p_cal),
                "Brier": float(brier_score_loss(y, p_cal)), "O_E": oe_ratio(y, p_cal),
                "CITL": calibration_in_the_large(y, p_cal),
                "Calibration_intercept": cal_intercept, "Calibration_slope": cal_slope,
                "target_specificity": 0.90, "threshold_AoU_CAL": threshold_90,
                "AoU_CAL_achieved_specificity": float(mr["achieved_cal_spec_90"]),
                **cm90,
            }
            if RUN_BOOTSTRAP:
                cis = bootstrap_metrics(y, p_cal, threshold_90)
                for metric, (lo, hi) in cis.items():
                    row[f"{metric}_ci_low"] = lo
                    row[f"{metric}_ci_high"] = hi
            perf_rows.append(row)

            for spec in [80, 85, 90, 95]:
                thr = float(mr[f"threshold_spec_{spec}"])
                cm = classification_metrics(y, p_cal, thr)
                policy_rows.append({
                    "analysis_group": group, "landmark": lm, "model_id": model_id, "algorithm": alg,
                    "target_specificity": spec / 100.0,
                    "threshold_AoU_CAL": thr,
                    "AoU_CAL_achieved_specificity": float(mr[f"achieved_cal_spec_{spec}"]),
                    **cm,
                })

            if group == "primary_biome_compatible":
                class_at_90[model_id] = (p_cal >= threshold_90).astype(int)

            print(
                f"  {label:<48} AUROC={safe_auc(y,p_cal):.4f} "
                f"AUPRC={safe_auprc(y,p_cal):.4f} "
                f"Sens@AoU90={cm90['Sensitivity']:.3f} Spec={cm90['Specificity']:.3f}"
            )

        # Paired patient-level reclassification at the frozen AoU 90% policy.
        primary_sequence = ["M1", "M2", "M3", "M4a" if lm in {"30d", "6m"} else "M4"]
        for earlier, later in zip(primary_sequence[:-1], primary_sequence[1:]):
            if earlier not in class_at_90 or later not in class_at_90:
                continue
            pe, pl = class_at_90[earlier], class_at_90[later]
            case_mask = y.to_numpy() == 1
            ctrl_mask = ~case_mask
            fn_to_tp = int(np.sum(case_mask & (pe == 0) & (pl == 1)))
            tp_to_fn = int(np.sum(case_mask & (pe == 1) & (pl == 0)))
            fp_to_tn = int(np.sum(ctrl_mask & (pe == 1) & (pl == 0)))
            tn_to_fp = int(np.sum(ctrl_mask & (pe == 0) & (pl == 1)))
            earlier_fn = int(np.sum(case_mask & (pe == 0)))
            later_fn = int(np.sum(case_mask & (pl == 0)))
            reclass_rows.append({
                "landmark": lm, "transition": f"{earlier}->{later}",
                "newly_detected_FN_to_TP": fn_to_tp,
                "lost_detection_TP_to_FN": tp_to_fn,
                "net_false_negative_reduction": earlier_fn - later_fn,
                "net_false_negative_reduction_check": fn_to_tp - tp_to_fn,
                "earlier_FN": earlier_fn, "later_FN": later_fn,
                "FP_to_TN": fp_to_tn, "TN_to_FP": tn_to_fp,
                "net_false_positive_change": tn_to_fp - fp_to_tn,
            })

        prob_path = OUT_DIR / f"biome_probabilities_{lm}_{RUN_DATE}.csv"
        prob_out.to_csv(prob_path, index=False)
        scored_by_landmark[lm] = prob_out
        print(f"  Saved dummy-ID probabilities: {prob_path}")

    # -------------------------------------------------------------------------
    # Save all summaries
    # -------------------------------------------------------------------------
    perf_df = pd.DataFrame(perf_rows)
    policy_df = pd.DataFrame(policy_rows)
    raw_cal_df = pd.DataFrame(raw_cal_rows)
    cohort_df = pd.DataFrame(cohort_rows)
    reclass_df = pd.DataFrame(reclass_rows)
    feature_df = pd.concat(all_feature_rows, ignore_index=True)
    domain_df = pd.concat(all_domain_rows, ignore_index=True)
    table1_df = pd.concat(table1_rows, ignore_index=True) if table1_rows else pd.DataFrame()
    table1_formatted_df = pd.concat(table1_formatted_rows, ignore_index=True) if table1_formatted_rows else pd.DataFrame()
    obs_df = pd.concat(obs_rows, ignore_index=True) if obs_rows else pd.DataFrame()
    obs_tertile_df = pd.concat(obs_tertile_rows, ignore_index=True) if obs_tertile_rows else pd.DataFrame()
    feature_outcome_df = pd.concat(all_feature_outcome_rows, ignore_index=True) if all_feature_outcome_rows else pd.DataFrame()
    domain_outcome_df = pd.concat(all_domain_outcome_rows, ignore_index=True) if all_domain_outcome_rows else pd.DataFrame()
    layer_missingness_df = pd.concat(all_layer_missingness_rows, ignore_index=True) if all_layer_missingness_rows else pd.DataFrame()
    coded_overall_df = pd.concat(coded_overall_rows, ignore_index=True) if coded_overall_rows else pd.DataFrame()
    suppfig3_long_df = pd.concat(suppfig3_long_rows, ignore_index=True) if suppfig3_long_rows else pd.DataFrame()
    suppfig3_gap_df = pd.concat(suppfig3_gap_rows, ignore_index=True) if suppfig3_gap_rows else pd.DataFrame()
    binary_normalization_df = pd.DataFrame(binary_normalization_rows)

    outputs = {
        "biome_external_cohort_summary": cohort_df,
        "biome_external_performance_90spec": perf_df,
        "biome_external_policy_80_85_90_95": policy_df,
        "biome_external_raw_vs_calibrated": raw_cal_df,
        "biome_external_case_reclassification_90spec": reclass_df,
        "biome_external_feature_coverage_by_feature": feature_df,
        "biome_external_feature_coverage_by_domain": domain_df,
        "biome_external_feature_coverage_by_feature_and_outcome": feature_outcome_df,
        "biome_external_feature_coverage_by_domain_and_outcome": domain_outcome_df,
        "biome_external_missingness_by_incremental_layer_and_outcome": layer_missingness_df,
        "biome_external_table1_all_landmarks_long": table1_df,
        "biome_external_table1_all_landmarks_formatted": table1_formatted_df,
        "biome_external_table1_1y_manuscript_ready": table1_formatted_df[table1_formatted_df["landmark"] == "1y"].copy() if not table1_formatted_df.empty else pd.DataFrame(),
        "biome_external_observation_density_all_landmarks": obs_df,
        "biome_external_observability_tertile_summary_all_landmarks": obs_tertile_df,
        "biome_external_coded_feature_prevalence_by_outcome_all_landmarks": coded_overall_df,
        "biome_external_suppfig3_prevalence_by_observability_tertile": suppfig3_long_df,
        "biome_external_suppfig3_case_control_gaps": suppfig3_gap_df,
        "biome_external_binary_feature_normalization_audit": binary_normalization_df,
    }
    written = []
    for stem, obj in outputs.items():
        if obj.empty:
            continue
        path = OUT_DIR / f"{stem}_{RUN_DATE}.csv"
        obj.to_csv(path, index=False)
        written.append(str(path))

    # -------------------------------------------------------------------------
    # Reproducibility metadata and one return ZIP
    # -------------------------------------------------------------------------
    bundle_path = OUT_DIR / f"biome_external_return_bundle_{RUN_DATE}.zip"
    run_manifest = {
        "run_date": RUN_DATE,
        "external_validation_rule": (
            "Frozen AoU models, calibrators, exact feature order, and AoU CAL-derived "
            "thresholds applied unchanged; no BioMe fitting/recalibration/threshold learning."
        ),
        "landmarks": list(BIOME_LANDMARK_PATHS),
        "n_models": int(len(model_manifest)),
        "n_model_objects": int(len(model_inventory_df)),
        "model_directory_used": str(model_dir),
        "model_zip_name": MODEL_ZIP_NAME,
        "bootstrap_run_on_biome": bool(RUN_BOOTSTRAP),
        "medication_binary_missingness_rule": (
            "Structural NA values in medication ever/recent flags caused by participants "
            "with no medication-source row are coded to 0 = no recorded exposure; "
            "malformed nonmissing values remain a hard failure."
        ),
        "return_bundle": str(bundle_path),
        "return_bundle_contains_real_patient_ids": False,
        "python": sys.version,
        "python_environment_actual": environment_actual,
        "python_environment_expected": EXPECTED_ENV,
    }
    manifest_path = OUT_DIR / f"biome_external_run_manifest_{RUN_DATE}.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2))
    written.append(str(manifest_path))
    written.append(str(model_inventory_path))

    # Include the exact AoU manifests/thresholds and requirements when present.
    support_paths = [FEATURE_MANIFEST_PATH, MODEL_MANIFEST_PATH, THRESHOLD_TABLE_PATH, REQUIREMENTS_PATH]
    support_paths = [p for p in support_paths if p.exists()]

    # Include return-safe phenotype/curation QC generated by 07B when present.
    r_qc_candidates = [
        BASE_DIR / "biome_landmark_cohort_qc_20260823.csv",
        BASE_DIR / "biome_dkd_outcome_source_qc_20260823.csv",
        BASE_DIR / "biome_followup_qc_20260823.csv",
        BASE_DIR / "biome_landmark_followup_distribution_20260823.csv",
        BASE_DIR / "biome_required_feature_coverage_20260823.csv",
        BASE_DIR / "biome_general_ckd_audit_20260823.csv",
        BASE_DIR / "biome_t2d_evidence_source_qc_20260823.csv",
        BASE_DIR / "biome_renal_persistence_source_qc_20260823.csv",
        BASE_DIR / "biome_uacr_pairing_qc_20260823.csv",
        BASE_DIR / "biome_post_t2d_computed_egfr_surveillance_qc_20260823.csv",
        BASE_DIR / "biome_prs_coverage_qc_20260823.csv",
        BASE_DIR / "biome_extraction_diag_20260823.txt",
        BASE_DIR / "session_info_biome_20260823.txt",
        BASE_DIR / "session_info_landmark.txt",
    ]
    r_qc_paths = [p for p in r_qc_candidates if p.exists()]

    probability_paths = [
        OUT_DIR / f"biome_probabilities_{lm}_{RUN_DATE}.csv"
        for lm in BIOME_LANDMARK_PATHS
    ]

    # Environment record and the scorer itself are also safe/useful for audit.
    audit_paths = [env_path]
    try:
        script_path = Path(__file__).resolve()
        if script_path.exists():
            audit_paths.append(script_path)
    except Exception:
        pass

    bundle_members = []
    seen = set()
    for p in [Path(x) for x in written] + probability_paths + support_paths + r_qc_paths + audit_paths:
        if p.exists() and p.resolve() not in seen:
            bundle_members.append(p)
            seen.add(p.resolve())

    # File inventory for the returned package.
    inventory_rows = []
    for p in bundle_members:
        inventory_rows.append({
            "filename": p.name,
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        })
    bundle_inventory_path = OUT_DIR / f"biome_external_return_file_inventory_{RUN_DATE}.csv"
    pd.DataFrame(inventory_rows).to_csv(bundle_inventory_path, index=False)
    bundle_members.append(bundle_inventory_path)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_members:
            zf.write(path, arcname=path.name)

    # Verify the ZIP itself before telling the collaborator it is complete.
    with zipfile.ZipFile(bundle_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Return ZIP integrity check failed at member: {bad}")
        zip_names = set(zf.namelist())
    required_return_names = {
        f"biome_probabilities_{lm}_{RUN_DATE}.csv" for lm in BIOME_LANDMARK_PATHS
    } | {
        f"biome_external_performance_90spec_{RUN_DATE}.csv",
        f"biome_external_policy_80_85_90_95_{RUN_DATE}.csv",
        f"biome_external_run_manifest_{RUN_DATE}.json",
        f"biome_external_model_inventory_{RUN_DATE}.csv",
        f"biome_external_binary_feature_normalization_audit_{RUN_DATE}.csv",
        f"biome_external_return_file_inventory_{RUN_DATE}.csv",
    }
    missing_return = sorted(required_return_names - zip_names)
    if missing_return:
        raise RuntimeError(f"Return ZIP is missing required output(s): {missing_return}")

    banner("RUN COMPLETE")
    print(perf_df[[
        "analysis_group", "landmark", "model_id", "algorithm", "n_features",
        "AUROC", "AUPRC", "Brier", "O_E", "CITL", "Calibration_slope",
        "Sensitivity", "Specificity", "PPV", "NPV", "TP", "FP", "TN", "FN",
    ]].to_string(index=False))
    print("\nReturn this single ZIP file:")
    print(bundle_path)
    print("It contains dummy IDs only.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        err_text = (
            "FATAL ERROR — BioMe validation stopped before silent substitution could occur.\n\n"
            + str(exc) + "\n\n" + traceback.format_exc()
        )
        print("\n" + err_text)
        try:
            BASE_DIR.mkdir(parents=True, exist_ok=True)
            err_path = BASE_DIR / f"biome_external_validation_ERROR_{RUN_DATE}.txt"
            err_path.write_text(err_text)
            print(f"Error report written to: {err_path}")
        except Exception:
            pass
        sys.exit(1)
