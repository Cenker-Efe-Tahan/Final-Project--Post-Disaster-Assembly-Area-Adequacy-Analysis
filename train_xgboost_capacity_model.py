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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "Train_Dataset.csv"
TEST_PATH = BASE_DIR / "Test_Dataset.csv"
OUT_DIR = BASE_DIR / "xgboost_capacity_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_THRESHOLD_M2_PER_PERSON = 1.5
RANDOM_STATE = 42

DERIVED_OR_TARGET_COLUMNS = [
    "KISI_BASI_M2",
    "TARGET_AREA",
    "AREA_DEFICIT",
    "VULNERABILITY_RATIO",
    "RISK_LABEL",
    "RISKLI_MI",
]
NUMERIC_FEATURES = ["NUFUS", "ALAN_M2", "LST_C", "NDVI", "NDBI"]
CATEGORICAL_FEATURES = ["ILCE"]
ID_COLUMNS = ["ILCE", "MAHALLE"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Risk/yetersizlik tanımı: AFAD minimumu olarak kullanılan 1.5 m²/kişi altında kalan mahalleler.
    out["RISKLI_MI"] = (out["KISI_BASI_M2"] < TARGET_THRESHOLD_M2_PER_PERSON).astype(int)
    return out


def build_model(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_risky_class": float(precision_score(y_true, pred, zero_division=0)),
        "recall_risky_class": float(recall_score(y_true, pred, zero_division=0)),
        "f1_risky_class": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "confusion_matrix_labels": ["0=Yeterli", "1=Riskli/Yetersiz"],
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["Yeterli", "Riskli/Yetersiz"],
            zero_division=0,
            output_dict=True,
        ),
    }


def get_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    preprocessor = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_plots(metrics: dict, fi: pd.DataFrame) -> None:
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Yeterli", "Riskli"])
    ax.set_yticks([0, 1], labels=["Yeterli", "Riskli"])
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("XGBoost Confusion Matrix - Test Seti")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
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
    train_raw, test_raw = load_data()
    train_df = add_target(train_raw)
    test_df = add_target(test_raw)

    missing_features = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in train_df.columns]
    if missing_features:
        raise ValueError(f"Eksik feature sütunları: {missing_features}")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["RISKLI_MI"]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["RISKLI_MI"]

    positives = int(y_train.sum())
    negatives = int((1 - y_train).sum())
    scale_pos_weight = negatives / positives if positives else 1.0

    pipe = build_model(scale_pos_weight=scale_pos_weight)
    pipe.fit(X_train, y_train)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    metrics = evaluate(y_test, test_proba, threshold=0.5)
    metrics.update({
        "target_definition": f"RISKLI_MI = 1 if KISI_BASI_M2 < {TARGET_THRESHOLD_M2_PER_PERSON}, else 0",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_risky_count": positives,
        "train_sufficient_count": negatives,
        "test_risky_count": int(y_test.sum()),
        "test_sufficient_count": int((1 - y_test).sum()),
        "features_used": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "excluded_columns_to_prevent_target_leakage": [c for c in DERIVED_OR_TARGET_COLUMNS if c in train_df.columns],
    })

    predictions = test_df[ID_COLUMNS + [
        "NUFUS", "ALAN_M2", "LST_C", "NDVI", "NDBI", "KISI_BASI_M2",
        "AREA_DEFICIT", "VULNERABILITY_RATIO", "RISKLI_MI"
    ]].copy()
    predictions["RISK_OLASILIGI"] = test_proba
    predictions["TAHMIN_RISKLI_MI"] = test_pred
    predictions["TAHMIN_ETIKETI"] = np.where(test_pred == 1, "Riskli/Yetersiz", "Yeterli")
    predictions["GERCEK_ETIKET"] = np.where(predictions["RISKLI_MI"] == 1, "Riskli/Yetersiz", "Yeterli")
    predictions["DOGRU_TAHMIN"] = predictions["RISKLI_MI"] == predictions["TAHMIN_RISKLI_MI"]
    predictions = predictions.sort_values("RISK_OLASILIGI", ascending=False).reset_index(drop=True)

    fi = get_feature_importance(pipe)

    joblib.dump(pipe, OUT_DIR / "xgboost_toplanma_alani_risk_model.joblib")
    predictions.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    save_plots(metrics, fi)

    report_lines = [
        "XGBoost Toplanma Alanı Yeterlilik/Risk Modeli",
        "=" * 56,
        f"Hedef: {metrics['target_definition']}",
        f"Kullanılan feature'lar: {', '.join(metrics['features_used'])}",
        f"Train: {metrics['train_rows']} satır | Riskli: {metrics['train_risky_count']} | Yeterli: {metrics['train_sufficient_count']}",
        f"Test:  {metrics['test_rows']} satır | Riskli: {metrics['test_risky_count']} | Yeterli: {metrics['test_sufficient_count']}",
        "",
        "Test metrikleri:",
        f"- Accuracy:          {metrics['accuracy']:.4f}",
        f"- Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Precision Riskli:  {metrics['precision_risky_class']:.4f}",
        f"- Recall Riskli:     {metrics['recall_risky_class']:.4f}",
        f"- F1 Riskli:         {metrics['f1_risky_class']:.4f}",
        f"- ROC-AUC:           {metrics['roc_auc']:.4f}",
        f"- Avg Precision:     {metrics['average_precision']:.4f}",
        "",
        "Confusion Matrix [[TN, FP], [FN, TP]]:",
        str(metrics["confusion_matrix"]),
        "",
        "En önemli 10 feature:",
        fi.head(10).to_string(index=False),
        "",
        "Not: KISI_BASI_M2, TARGET_AREA, AREA_DEFICIT ve VULNERABILITY_RATIO hedefi doğrudan türettiği için feature olarak kullanılmadı.",
    ]
    (OUT_DIR / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
