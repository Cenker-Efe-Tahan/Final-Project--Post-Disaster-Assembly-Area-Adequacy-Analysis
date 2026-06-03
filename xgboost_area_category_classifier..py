from __future__ import annotations

from pathlib import Path
import json
import re
import shutil
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "output"
OUTPUT_DIR = BASE_DIR / "xgboost_area_category_outputs"

RANDOM_STATE = 42
TARGET = "AREA_CATEGORY"

# Feature selection is intentionally identical to the reference Random Forest code.
# MAHALLE is kept only as an identifier in the prediction export, not as a model input.
FEATURES = ["ILCE", "NUFUS", "LST_C", "NDVI", "NDBI"]
FEATURES_ENCODED = ["ILCE_encoded", "NUFUS", "LST_C", "NDVI", "NDBI"]
ID_COLUMNS = ["ILCE", "MAHALLE"]

COPY_SUFFIX_RE = re.compile(r"\s*\(\d+\)(?=\.[^.]+$)")


def canonical_name(filename: str) -> str:
    """Convert names like Train_Dataset(3).csv to Train_Dataset.csv."""
    return COPY_SUFFIX_RE.sub("", filename)


def find_dataset(expected_filename: str) -> Path:
    """Find the dataset inside the root-level output folder while ignoring upload copy suffixes."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    exact_path = DATA_DIR / expected_filename
    if exact_path.exists():
        return exact_path

    matches = [
        path
        for path in DATA_DIR.glob("*.csv")
        if canonical_name(path.name) == expected_filename
    ]
    if not matches:
        stem = Path(expected_filename).stem
        raise FileNotFoundError(
            f"{expected_filename} was not found in {DATA_DIR}. Accepted examples: "
            f"{expected_filename}, {stem}(1).csv, {stem}(2).csv, {stem}(3).csv."
        )

    matches.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    selected = matches[0]
    print(f"Info: Using {selected.name} as {expected_filename}; upload copy suffix ignored.")
    return selected


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    train_path = find_dataset("Train_Dataset.csv")
    test_path = find_dataset("Test_Dataset.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df, train_path, test_path


def validate_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    required_columns = FEATURES + [TARGET]

    missing_train = [column for column in required_columns if column not in train_df.columns]
    missing_test = [column for column in required_columns if column not in test_df.columns]

    if missing_train:
        raise ValueError(f"Missing columns in training data: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test data: {missing_test}")

    train_classes = set(train_df[TARGET].dropna().astype(str).unique())
    test_classes = set(test_df[TARGET].dropna().astype(str).unique())
    unseen_test_classes = sorted(test_classes - train_classes)
    if unseen_test_classes:
        raise ValueError(
            "The test set contains classes that are not present in the training set: "
            f"{unseen_test_classes}"
        )


def encode_ilce_like_reference(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Encode ILCE exactly like the reference code: fit on combined train + test ILCE values."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    le_ilce = LabelEncoder()
    all_ilce = pd.concat([train_encoded["ILCE"], test_encoded["ILCE"]], axis=0).astype(str)
    le_ilce.fit(all_ilce)

    train_encoded["ILCE_encoded"] = le_ilce.transform(train_encoded["ILCE"].astype(str))
    test_encoded["ILCE_encoded"] = le_ilce.transform(test_encoded["ILCE"].astype(str))

    return train_encoded, test_encoded, le_ilce


def build_xgboost_classifier(num_classes: int) -> XGBClassifier:
    params = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
        "objective": "multi:softprob" if num_classes > 2 else "binary:logistic",
    }
    if num_classes > 2:
        params["num_class"] = num_classes
    return XGBClassifier(**params)


def save_confusion_matrix_plot(cm: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_title("Machine Learning Performance: Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Category", fontsize=12, fontweight="bold")

    threshold = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(cm[row_idx, col_idx]),
                ha="center",
                va="center",
                color="white" if cm[row_idx, col_idx] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "1_ML_Confusion_Matrix.png", dpi=300)
    plt.close(fig)


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    plot_df = importance_df.sort_values(by="Importance (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(plot_df["Feature"], plot_df["Importance (%)"])

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Impact Weight on Model Decisions (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Features", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_ML_Feature_Importance.png", dpi=300)
    plt.close(fig)


def main() -> None:
    reset_output_dir()

    train_df, test_df, train_path, test_path = load_data()
    validate_columns(train_df, test_df)

    train_df, test_df, le_ilce = encode_ilce_like_reference(train_df, test_df)

    X_train = train_df.loc[:, FEATURES_ENCODED].copy()
    X_test = test_df.loc[:, FEATURES_ENCODED].copy()

    if list(X_train.columns) != FEATURES_ENCODED or list(X_test.columns) != FEATURES_ENCODED:
        raise RuntimeError("Model input columns do not match the intended encoded feature list.")

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train_df[TARGET].astype(str))
    y_test = target_encoder.transform(test_df[TARGET].astype(str))
    labels = list(target_encoder.classes_)

    model = build_xgboost_classifier(num_classes=len(labels))

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    actual_labels = target_encoder.inverse_transform(y_test)
    predicted_labels = target_encoder.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, target_names=labels, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=labels, zero_division=0, output_dict=True)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual - {label}" for label in labels],
        columns=[f"Predicted - {label}" for label in labels],
    )

    feature_importance = model.feature_importances_ * 100
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance (%)": feature_importance,
        }
    ).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)

    prediction_columns = [column for column in ID_COLUMNS + FEATURES_ENCODED + [TARGET] if column in test_df.columns]
    predictions_df = test_df.loc[:, prediction_columns].copy()
    predictions_df["Actual Category"] = actual_labels
    predictions_df["Predicted Category"] = predicted_labels
    predictions_df["Correct Prediction"] = predictions_df["Actual Category"] == predictions_df["Predicted Category"]
    predictions_df["Prediction Confidence"] = y_proba.max(axis=1)

    probability_df = pd.DataFrame(y_proba, columns=[f"Probability - {label}" for label in labels])
    predictions_df = pd.concat([predictions_df, probability_df], axis=1)
    predictions_df = predictions_df.sort_values(
        ["Correct Prediction", "Prediction Confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)

    metrics = {
        "accuracy": float(accuracy),
        "target": TARGET,
        "features_utilized": FEATURES,
        "encoded_features_used_by_model": FEATURES_ENCODED,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_viable_neighborhoods": int(len(train_df) + len(test_df)),
        "training_file": train_path.name,
        "test_file": test_path.name,
        "class_labels": labels,
        "class_distribution_train": train_df[TARGET].value_counts().sort_index().to_dict(),
        "class_distribution_test": test_df[TARGET].value_counts().sort_index().to_dict(),
        "ilce_encoder_classes": list(le_ilce.classes_),
    }

    model_artifact = {
        "model": model,
        "ilce_encoder": le_ilce,
        "target_encoder": target_encoder,
        "features_utilized": FEATURES,
        "encoded_features_used_by_model": FEATURES_ENCODED,
        "target": TARGET,
        "labels": labels,
    }

    joblib.dump(model_artifact, OUTPUT_DIR / "xgboost_area_category_classifier.joblib")
    predictions_df.to_csv(OUTPUT_DIR / "test_classification_predictions.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv",
        encoding="utf-8-sig",
    )

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "model_input_columns.json", "w", encoding="utf-8") as file:
        json.dump({"X_train_columns": list(X_train.columns), "X_test_columns": list(X_test.columns)}, file, indent=2)

    save_confusion_matrix_plot(cm, labels)
    save_feature_importance_plot(importance_df)

    report_lines = [
        "XGBoost AREA_CATEGORY Classification Model",
        "=" * 64,
        f"Target: {TARGET}",
        f"Training file: {train_path.name}",
        f"Test file:     {test_path.name}",
        f"Total viable neighborhoods for training: {len(train_df) + len(test_df)}",
        f"Features utilized: {', '.join(FEATURES)}",
        f"Exact model input columns: {', '.join(FEATURES_ENCODED)}",
        f"Loaded pre-split data: {len(X_train)} Train | {len(X_test)} Test",
        "",
        "MODEL PERFORMANCE REPORT",
        "=" * 64,
        f"Overall Accuracy : {accuracy:.2%}",
        "=" * 64,
        "",
        report_text,
        "=" * 64,
        "",
        "Confusion matrix:",
        cm_df.to_string(),
        "",
        "Feature importance:",
        importance_df.to_string(index=False),
        "",
        "Findings:",
        "- The model input matrix contains exactly five columns: ILCE_encoded, NUFUS, LST_C, NDVI, and NDBI.",
        "- ILCE is encoded once with LabelEncoder fitted on combined train and test ILCE values, matching the reference code.",
        "- MAHALLE is used only as an identifier in the prediction export and is not included in X_train or X_test.",
        "- Parenthesized upload copy suffixes such as Train_Dataset(3).csv or Test_Dataset(3).csv are ignored.",
    ]

    (OUTPUT_DIR / "model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"Confusion Matrix plot saved as '{OUTPUT_DIR / '1_ML_Confusion_Matrix.png'}'")
    print(f"Feature Importance plot saved as '{OUTPUT_DIR / '2_ML_Feature_Importance.png'}'")


if __name__ == "__main__":
    main()
