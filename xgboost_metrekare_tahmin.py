from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "Train_Dataset.csv"
TEST_PATH = BASE_DIR / "Test_Dataset.csv"
OUT_DIR = BASE_DIR / "xgboost_metrekare_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COLUMN = "KISI_BASI_M2"

DERIVED_OR_TARGET_COLUMNS = [
    "KISI_BASI_M2",
    "TARGET_AREA",
    "AREA_DEFICIT",
    "VULNERABILITY_RATIO",
    "RISK_LABEL",
    "RISKLI_MI",
]

# Kullanılan veriler/feature seti değiştirilmedi.
NUMERIC_FEATURES = ["NUFUS", "ALAN_M2", "LST_C", "NDVI", "NDBI"]
CATEGORICAL_FEATURES = ["ILCE"]
ID_COLUMNS = ["ILCE", "MAHALLE"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.05,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    nonzero_mask = y_true_arr != 0
    if nonzero_mask.any():
        mape = float(mean_absolute_percentage_error(y_true_arr[nonzero_mask], y_pred_arr[nonzero_mask]) * 100)
    else:
        mape = None

    return {
        "rmse": rmse,
        "mape_percent": mape,
        "r2": r2,
        "mape_zero_actual_count_excluded": int((~nonzero_mask).sum()),
    }


def get_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_plots(y_true: pd.Series, y_pred: np.ndarray, fi: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.75)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Gerçek Kişi Başı m²")
    ax.set_ylabel("Tahmin Kişi Başı m²")
    ax.set_title("Gerçek vs Tahmin - Test Seti")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.75)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Tahmin Kişi Başı m²")
    ax.set_ylabel("Hata / Residual")
    ax.set_title("Residual Plot - Test Seti")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "residual_plot.png", dpi=200)
    plt.close(fig)

    top_fi = fi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_fi["feature"], top_fi["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance - Top 15")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    train_df, test_df = load_data()

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_train_columns = [c for c in required_columns if c not in train_df.columns]
    missing_test_columns = [c for c in required_columns if c not in test_df.columns]

    if missing_train_columns:
        raise ValueError(f"Train setinde eksik sütunlar: {missing_train_columns}")
    if missing_test_columns:
        raise ValueError(f"Test setinde eksik sütunlar: {missing_test_columns}")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET_COLUMN]

    pipe = build_model()
    pipe.fit(X_train, y_train)

    test_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, test_pred)
    metrics.update({
        "target_definition": f"{TARGET_COLUMN} regresyon tahmini",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_columns_to_prevent_target_leakage": [
            c for c in DERIVED_OR_TARGET_COLUMNS if c in train_df.columns and c not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ],
    })

    base_prediction_columns = [c for c in ID_COLUMNS + NUMERIC_FEATURES + [TARGET_COLUMN] if c in test_df.columns]
    predictions = test_df[base_prediction_columns].copy()
    predictions["TAHMIN_KISI_BASI_M2"] = test_pred
    predictions["HATA"] = predictions[TARGET_COLUMN] - predictions["TAHMIN_KISI_BASI_M2"]
    predictions["ABS_HATA"] = predictions["HATA"].abs()
    predictions["APE_PERCENT"] = np.where(
        predictions[TARGET_COLUMN] != 0,
        predictions["ABS_HATA"] / predictions[TARGET_COLUMN].abs() * 100,
        np.nan,
    )
    predictions = predictions.sort_values("ABS_HATA", ascending=False).reset_index(drop=True)

    fi = get_feature_importance(pipe)

    joblib.dump(pipe, OUT_DIR / "xgboost_metrekare_tahmin_model.joblib")
    predictions.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_plots(y_test, test_pred, fi)

    mape_text = "Hesaplanamadı" if metrics["mape_percent"] is None else f"{metrics['mape_percent']:.2f}%"
    report_lines = [
        "XGBoost Metrekare Tahmin Modeli",
        "=" * 56,
        f"Hedef: {metrics['target_definition']}",
        f"Kullanılan feature'lar: {', '.join(metrics['features_used'])}",
        f"Train: {metrics['train_rows']} satır",
        f"Test:  {metrics['test_rows']} satır",
        "",
        "Test metrikleri:",
        f"- RMSE: {metrics['rmse']:.4f}",
        f"- MAPE: {mape_text}",
        f"- R²:   {metrics['r2']:.4f}",
    ]

    if metrics["mape_zero_actual_count_excluded"] > 0:
        report_lines.append(
            f"Not: MAPE hesaplanırken gerçek değeri 0 olan {metrics['mape_zero_actual_count_excluded']} satır dışarıda bırakıldı."
        )

    report_lines.extend([
        "",
        "En önemli 10 feature:",
        fi.head(10).to_string(index=False),
        "",
        "Not: Kullanılan feature listesi değiştirilmedi. Model sınıflandırma yerine KISI_BASI_M2 regresyon tahmini yapar.",
    ])

    (OUT_DIR / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
