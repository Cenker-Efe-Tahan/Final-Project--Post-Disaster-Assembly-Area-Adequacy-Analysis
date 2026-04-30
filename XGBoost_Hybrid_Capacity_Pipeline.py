from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import openpyxl
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None

BASE_DIR = Path(__file__).resolve().parent
PROJECTION_FILE = BASE_DIR / "output" / "neighbourhood_2026_2027_projection.csv"
MATCH_REPORT_FILE = BASE_DIR / "Eslesme_Raporu.xlsx"
AREA_FILE = BASE_DIR / "Acil_Toplanma_Alanları.xlsx"
OUTPUT_DIR = BASE_DIR / "output_xgboost_hybrid"

TARGET_PER_CAPITA_M2 = 1.5


# ==========================================
# 1. NORMALIZATION HELPERS

def fix_turkish_letters(text):
    if pd.isna(text):
        return text
    return str(text).upper().translate(str.maketrans("ÇĞİÖŞÜ", "CGIOSU")).strip()


def normalize_neighborhood_for_projection(text: str) -> str:
    if pd.isna(text):
        return ""
    text = fix_turkish_letters(text)
    text = text.replace("MAH.", "")
    text = text.replace("MAHALLESI", "")
    text = text.replace(" ", "")
    return text


def normalize_key(text: str) -> str:
    if pd.isna(text):
        return ""
    return fix_turkish_letters(text).replace(" ", "")


# ==========================================
# 2. INPUT LOADERS

def smart_excel_parser(file_path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    rows = [row for row in sheet.iter_rows(min_row=2, values_only=True)]
    df = pd.DataFrame(rows)
    df = df[[2, 4, 8]]
    df.columns = ["ILCE", "MAHALLE", "ALAN_M2"]
    return df


def clean_area(val) -> float:
    if pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val_str = val.strip().replace(".", "").replace(",", ".")
        try:
            return float(val_str)
        except ValueError:
            return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        if (val != int(val)) or (0 < val < 100):
            return float(val) * 1000
        return float(val)
    return 0.0


def load_projection_data() -> pd.DataFrame:
    df = pd.read_csv(PROJECTION_FILE)
    df["ILCE_ORJ"] = df["ILCE"]
    df["MAHALLE_ORJ"] = df["MAHALLE"]
    df["ILCE_KEY"] = df["ILCE"].apply(normalize_key)
    df["MAHALLE_KEY"] = df["MAHALLE"].apply(normalize_neighborhood_for_projection)
    return df


def load_match_report() -> pd.DataFrame:
    df = pd.read_excel(MATCH_REPORT_FILE)
    df["ILCE_KEY"] = df["ILCE"].apply(normalize_key)
    df["MAHALLE_KEY"] = df["MAHALLE"].apply(normalize_key)
    df["AFAD_MAHALLE_KEY"] = df["AFAD_MAHALLE"].apply(normalize_key)
    return df[["ILCE_KEY", "MAHALLE_KEY", "AFAD_MAHALLE", "AFAD_MAHALLE_KEY", "ESLESME_TURU"]]


def load_area_data() -> pd.DataFrame:
    df = smart_excel_parser(AREA_FILE)
    df["ALAN_M2"] = df["ALAN_M2"].apply(clean_area)
    df["ILCE_KEY"] = df["ILCE"].apply(normalize_key)
    df["AFAD_MAHALLE_KEY"] = df["MAHALLE"].apply(normalize_key)
    grouped = (
        df.groupby(["ILCE_KEY", "AFAD_MAHALLE_KEY"], as_index=False)["ALAN_M2"]
        .sum()
    )
    return grouped


# ==========================================
# 3. MASTER DATASET

def build_master_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    projection_df = load_projection_data()
    match_df = load_match_report()
    area_df = load_area_data()

    merged = projection_df.merge(
        match_df,
        on=["ILCE_KEY", "MAHALLE_KEY"],
        how="left",
    )

    unmatched = merged[merged["AFAD_MAHALLE_KEY"].isna()].copy()
    matched = merged[merged["AFAD_MAHALLE_KEY"].notna()].copy()

    matched = matched.merge(
        area_df,
        on=["ILCE_KEY", "AFAD_MAHALLE_KEY"],
        how="left",
    )

    matched = matched[matched["ALAN_M2"].notna()].copy()
    matched["ILCE"] = matched["ILCE_KEY"]
    matched["MAHALLE"] = matched["MAHALLE_KEY"]

    matched["KISI_BASI_M2_2025"] = matched["ALAN_M2"] / matched["NUFUS_2025"]
    matched["KISI_BASI_M2_2026"] = matched["ALAN_M2"] / matched["TAHMINI_NUFUS_2026"]
    matched["KISI_BASI_M2_2027"] = matched["ALAN_M2"] / matched["TAHMINI_NUFUS_2027"]

    matched["NEIGHBORHOOD_ID"] = matched["ILCE"] + "__" + matched["MAHALLE"]
    return matched, unmatched


# ==========================================
# 4. FEATURE ENGINEERING

def add_district_context(df: pd.DataFrame, current_pop_col: str, per_capita_col: str) -> pd.DataFrame:
    out = df.copy()
    out["IS_CURRENT_RISK"] = (out[per_capita_col] < TARGET_PER_CAPITA_M2).astype(int)

    district_total_pop = out.groupby("ILCE")[current_pop_col].transform("sum")
    district_mean_area = out.groupby("ILCE")[per_capita_col].transform("mean")
    district_risk_ratio = out.groupby("ILCE")["IS_CURRENT_RISK"].transform("mean")
    district_neighborhood_count = out.groupby("ILCE")["MAHALLE"].transform("count")

    out["ILCE_TOPLAM_NUFUS"] = district_total_pop
    out["ILCE_ORTALAMA_KISI_BASI_ALAN"] = district_mean_area
    out["ILCE_RISK_ORANI"] = district_risk_ratio
    out["ILCE_MAHALLE_SAYISI"] = district_neighborhood_count

    out["CURRENT_POP"] = out[current_pop_col]
    out["CURRENT_PER_CAPITA_M2"] = out[per_capita_col]
    out["CURRENT_TARGET_AREA_M2"] = out[current_pop_col] * TARGET_PER_CAPITA_M2
    out["CURRENT_EXTRA_AREA_GAP_M2"] = (out["CURRENT_TARGET_AREA_M2"] - out["ALAN_M2"]).clip(lower=0)
    out["LOG_CURRENT_POP"] = np.log1p(out["CURRENT_POP"])
    out["LOG_ALAN_M2"] = np.log1p(out["ALAN_M2"])
    return out


def build_training_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    train_2025 = add_district_context(master_df, "NUFUS_2025", "KISI_BASI_M2_2025")
    train_2025["CURRENT_YEAR"] = 2025
    train_2025["NEXT_YEAR"] = 2026
    train_2025["TARGET_NEXT_POP"] = train_2025["TAHMINI_NUFUS_2026"]

    train_2026 = add_district_context(master_df, "TAHMINI_NUFUS_2026", "KISI_BASI_M2_2026")
    train_2026["CURRENT_YEAR"] = 2026
    train_2026["NEXT_YEAR"] = 2027
    train_2026["TARGET_NEXT_POP"] = train_2026["TAHMINI_NUFUS_2027"]

    stacked = pd.concat([train_2025, train_2026], ignore_index=True)
    return stacked


FEATURE_COLUMNS = [
    "CURRENT_YEAR",
    "CURRENT_POP",
    "LOG_CURRENT_POP",
    "ARTIS_HIZI_BINDE",
    "ALAN_M2",
    "LOG_ALAN_M2",
    "CURRENT_PER_CAPITA_M2",
    "CURRENT_TARGET_AREA_M2",
    "CURRENT_EXTRA_AREA_GAP_M2",
    "IS_CURRENT_RISK",
    "ILCE_TOPLAM_NUFUS",
    "ILCE_ORTALAMA_KISI_BASI_ALAN",
    "ILCE_RISK_ORANI",
    "ILCE_MAHALLE_SAYISI",
    "ILCE",
]

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c != "ILCE"]
CATEGORICAL_FEATURES = ["ILCE"]


def build_model_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ==========================================
# 5. TRAIN / EVALUATE / REFIT

def evaluate_model(train_df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = train_df[FEATURE_COLUMNS].copy()
    y = train_df["TARGET_NEXT_POP"].copy()
    groups = train_df["NEIGHBORHOOD_ID"].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)
    pred_test = pipeline.predict(X_test)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_neighborhoods": int(groups.iloc[train_idx].nunique()),
        "test_neighborhoods": int(groups.iloc[test_idx].nunique()),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, pred_test))),
        "r2": float(r2_score(y_test, pred_test)),
    }

    final_pipeline = build_model_pipeline()
    final_pipeline.fit(X, y)
    return final_pipeline, metrics


# ==========================================
# 6. RECURSIVE FUTURE POPULATION PREDICTION

def make_prediction_frame(master_df: pd.DataFrame, current_year: int, current_pop_col: str, per_capita_col: str) -> pd.DataFrame:
    frame = add_district_context(master_df, current_pop_col, per_capita_col)
    frame["CURRENT_YEAR"] = current_year
    return frame


def predict_future_population(master_df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    result = master_df.copy()

    frame_2025 = make_prediction_frame(result, 2025, "NUFUS_2025", "KISI_BASI_M2_2025")
    result["XGB_TAHMINI_NUFUS_2026"] = (
        np.round(pipeline.predict(frame_2025[FEATURE_COLUMNS])).astype(int)
    )
    result["XGB_TAHMINI_NUFUS_2026"] = result["XGB_TAHMINI_NUFUS_2026"].clip(lower=0)
    result["XGB_KISI_BASI_M2_2026"] = result["ALAN_M2"] / result["XGB_TAHMINI_NUFUS_2026"].replace(0, np.nan)

    frame_2026 = result.copy()
    frame_2026["KISI_BASI_M2_MODEL_2026"] = result["ALAN_M2"] / result["XGB_TAHMINI_NUFUS_2026"].replace(0, np.nan)
    frame_2026 = make_prediction_frame(frame_2026, 2026, "XGB_TAHMINI_NUFUS_2026", "KISI_BASI_M2_MODEL_2026")

    result["XGB_TAHMINI_NUFUS_2027"] = (
        np.round(pipeline.predict(frame_2026[FEATURE_COLUMNS])).astype(int)
    )
    result["XGB_TAHMINI_NUFUS_2027"] = result["XGB_TAHMINI_NUFUS_2027"].clip(lower=0)
    result["XGB_KISI_BASI_M2_2027"] = result["ALAN_M2"] / result["XGB_TAHMINI_NUFUS_2027"].replace(0, np.nan)

    return result


# ==========================================
# 7. RULE-BASED CAPACITY CALCULATION

def apply_capacity_rule(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()

    out["HEDEF_ALAN_M2_2025"] = (out["NUFUS_2025"] * TARGET_PER_CAPITA_M2).round(2)
    out["HEDEF_ALAN_M2_XGB_2026"] = (out["XGB_TAHMINI_NUFUS_2026"] * TARGET_PER_CAPITA_M2).round(2)
    out["HEDEF_ALAN_M2_XGB_2027"] = (out["XGB_TAHMINI_NUFUS_2027"] * TARGET_PER_CAPITA_M2).round(2)

    out["EKLENMESI_GEREKEN_ALAN_M2_2025"] = (out["HEDEF_ALAN_M2_2025"] - out["ALAN_M2"]).clip(lower=0).round(2)
    out["EKLENMESI_GEREKEN_ALAN_M2_XGB_2026"] = (out["HEDEF_ALAN_M2_XGB_2026"] - out["ALAN_M2"]).clip(lower=0).round(2)
    out["EKLENMESI_GEREKEN_ALAN_M2_XGB_2027"] = (out["HEDEF_ALAN_M2_XGB_2027"] - out["ALAN_M2"]).clip(lower=0).round(2)

    out["XGB_RISK_2025"] = (out["KISI_BASI_M2_2025"] < TARGET_PER_CAPITA_M2).astype(int)
    out["XGB_RISK_2026"] = (out["XGB_KISI_BASI_M2_2026"] < TARGET_PER_CAPITA_M2).astype(int)
    out["XGB_RISK_2027"] = (out["XGB_KISI_BASI_M2_2027"] < TARGET_PER_CAPITA_M2).astype(int)

    return out


# ==========================================
# 8. EXPORTS

def export_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "FEATURE": feature_names,
            "IMPORTANCE": model.feature_importances_,
        }
    ).sort_values("IMPORTANCE", ascending=False)
    importance.to_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")
    return importance


def export_outputs(final_df: pd.DataFrame, unmatched_df: pd.DataFrame, metrics: dict, pipeline: Pipeline) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    columns_full = [
        "ILCE_ORJ",
        "MAHALLE_ORJ",
        "ALAN_M2",
        "NUFUS_2025",
        "TAHMINI_NUFUS_2026",
        "TAHMINI_NUFUS_2027",
        "XGB_TAHMINI_NUFUS_2026",
        "XGB_TAHMINI_NUFUS_2027",
        "KISI_BASI_M2_2025",
        "XGB_KISI_BASI_M2_2026",
        "XGB_KISI_BASI_M2_2027",
        "HEDEF_ALAN_M2_2025",
        "HEDEF_ALAN_M2_XGB_2026",
        "HEDEF_ALAN_M2_XGB_2027",
        "EKLENMESI_GEREKEN_ALAN_M2_2025",
        "EKLENMESI_GEREKEN_ALAN_M2_XGB_2026",
        "EKLENMESI_GEREKEN_ALAN_M2_XGB_2027",
        "XGB_RISK_2025",
        "XGB_RISK_2026",
        "XGB_RISK_2027",
        "ESLESME_TURU",
    ]

    final_sorted = final_df.sort_values(
        by=["EKLENMESI_GEREKEN_ALAN_M2_XGB_2027", "EKLENMESI_GEREKEN_ALAN_M2_XGB_2026"],
        ascending=False,
    ).reset_index(drop=True)

    full_path = OUTPUT_DIR / "xgb_population_capacity_projection_full.csv"
    final_sorted[columns_full].to_csv(full_path, index=False, encoding="utf-8-sig")

    risk_only = final_sorted[
        (final_sorted["XGB_RISK_2025"] == 1)
        | (final_sorted["XGB_RISK_2026"] == 1)
        | (final_sorted["XGB_RISK_2027"] == 1)
    ].copy()
    risk_path = OUTPUT_DIR / "xgb_population_capacity_projection_risk_only.csv"
    risk_only[columns_full].to_csv(risk_path, index=False, encoding="utf-8-sig")

    unmatched_out = unmatched_df[["ILCE_ORJ", "MAHALLE_ORJ"]].copy()
    unmatched_out.to_csv(OUTPUT_DIR / "unmatched_neighborhoods.csv", index=False, encoding="utf-8-sig")

    importance_df = export_feature_importance(pipeline)
    joblib.dump(pipeline, OUTPUT_DIR / "xgb_population_model.joblib")

    summary_lines = [
        "============================================================",
        "XGBOOST HYBRID POPULATION + CAPACITY PIPELINE SUMMARY",
        "============================================================",
        f"Matched neighborhoods used in model/capacity pipeline: {len(final_df)}",
        f"Unmatched neighborhoods excluded from area merge: {len(unmatched_df)}",
        f"Train rows: {metrics['train_rows']}",
        f"Test rows: {metrics['test_rows']}",
        f"Train neighborhoods: {metrics['train_neighborhoods']}",
        f"Test neighborhoods: {metrics['test_neighborhoods']}",
        "",
        "Held-out evaluation metrics",
        f"MAE  : {metrics['mae']:.4f}",
        f"RMSE : {metrics['rmse']:.4f}",
        f"R^2  : {metrics['r2']:.4f}",
        "",
        "Important note:",
        "The currently available 2026 and 2027 targets come from the existing projection file.",
        "So this script correctly implements the recommended hybrid pipeline, but for production",
        "the model should be retrained with real multi-year neighborhood population history.",
        "",
        "Top 10 model features",
    ]

    for _, row in importance_df.head(10).iterrows():
        summary_lines.append(f"- {row['FEATURE']}: {row['IMPORTANCE']:.6f}")

    top_2027 = risk_only[[
        "ILCE_ORJ", "MAHALLE_ORJ", "XGB_TAHMINI_NUFUS_2027", "EKLENMESI_GEREKEN_ALAN_M2_XGB_2027"
    ]].head(20)

    summary_lines.append("")
    summary_lines.append("Top 20 risk-side neighborhoods by 2027 extra area need")
    summary_lines.append(top_2027.to_string(index=False))

    (OUTPUT_DIR / "xgb_model_metrics_and_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    machine_summary = {
        "metrics": metrics,
        "matched_neighborhoods": int(len(final_df)),
        "unmatched_neighborhoods": int(len(unmatched_df)),
        "risk_neighborhoods_any_year": int(len(risk_only)),
        "top_2027_extra_area_total": float(risk_only["EKLENMESI_GEREKEN_ALAN_M2_XGB_2027"].head(20).sum()),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(machine_summary, indent=2), encoding="utf-8")


# ==========================================
# 9. MAIN

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROJECTION_FILE.exists():
        raise FileNotFoundError(f"Projection file not found: {PROJECTION_FILE}")
    if not MATCH_REPORT_FILE.exists():
        raise FileNotFoundError(f"Match report file not found: {MATCH_REPORT_FILE}")
    if not AREA_FILE.exists():
        raise FileNotFoundError(f"Area file not found: {AREA_FILE}")

    master_df, unmatched_df = build_master_dataset()
    train_df = build_training_dataset(master_df)

    pipeline, metrics = evaluate_model(train_df)
    predicted_df = predict_future_population(master_df, pipeline)
    final_df = apply_capacity_rule(predicted_df)

    export_outputs(final_df, unmatched_df, metrics, pipeline)

    print("[SUCCESS] XGBoost hybrid population-capacity pipeline completed.")
    print(f"Matched neighborhoods: {len(final_df)}")
    print(f"Unmatched neighborhoods: {len(unmatched_df)}")
    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
