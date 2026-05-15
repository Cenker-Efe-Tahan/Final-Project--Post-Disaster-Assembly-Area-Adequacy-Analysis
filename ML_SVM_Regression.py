import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
pd.options.mode.chained_assignment = None

output_dir = "ML_SVM-Charts"
os.makedirs(output_dir, exist_ok=True)

# Load pre-split frozen datasets for absolute fairness
train_df = pd.read_csv("output/Train_Dataset.csv")
test_df = pd.read_csv("output/Test_Dataset.csv")

# Feature Selection
features = ['NDVI', 'NDBI', 'LST_C']
target = 'VULNERABILITY_RATIO'
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

print()
print(f"Total viable neighborhoods for training: {len(train_df) + len(test_df)}")
print("Features utilized: Green Cover (NDVI), Built-up Index (NDBI), Surface Temp (LST_C)")
print(f"Loaded Pre-Split Data: {len(X_train)} Train | {len(X_test)} Test")

# ==========================================
# FEATURE SCALING (CRITICAL FOR SVM)
print("\nScaling features (essential for SVM kernel computations)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# HYPERPARAMETER TUNING: Find Optimal C
print("\nTuning hyperparameter 'C' via cross-validation...")
C_values = [0.1, 1, 10, 50, 100, 500, 1000]
cv_scores = []

for C in C_values:
    svm = SVR(kernel='rbf', C=C, gamma='scale', epsilon=0.1)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    print(f"C={C:7.1f} | CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Find optimal C
optimal_C = C_values[np.argmax(cv_scores)]
print(f"\n[OK] Optimal C: {optimal_C} (R2 Score: {max(cv_scores):.4f})")

# ==========================================
# TRAINING with Optimal C
print("\nTraining SVM Regressor with optimal parameters...")
svm_model = SVR(
    kernel='rbf',
    C=optimal_C,
    gamma='scale',
    epsilon=0.1
)
svm_model.fit(X_train_scaled, y_train)

# ==========================================
# PREDICTION & EVALUATION
y_pred = svm_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
mae = mean_absolute_error(y_test.values, y_pred)
r2 = r2_score(y_test.values, y_pred)
n_support_vectors = len(svm_model.support_)

print()
print("             MODEL PERFORMANCE REPORT             ")
print("==================================================")
print(f"Root Mean Square Error (RMSE) : {rmse:.2f} %")
print(f"Mean Absolute Error (MAE)     : {mae:.2f} %")
print(f"R-Squared (R2) Score          : {r2:.3f}")
print(f"Optimal C Parameter           : {optimal_C}")
print(f"Support Vectors               : {n_support_vectors}")
print("==================================================\n")

# ==========================================
# VISUALIZATIONS

# 1. C-value Tuning Curve
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, cv_scores, marker='o', linestyle='-', linewidth=2, markersize=6, color='steelblue')
plt.axvline(x=optimal_C, color='red', linestyle='--', linewidth=2, label=f'Optimal C={optimal_C}')
plt.xlabel('Regularization Parameter C (log scale)', fontsize=11, fontweight='bold')
plt.ylabel('Cross-Validation R² Score', fontsize=11, fontweight='bold')
plt.title('SVM: Hyperparameter Tuning - Optimal C Selection', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_SVM_C_Tuning_Curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test.values, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.title(f'SVM Predictions vs Actual (R² = {r2:.3f})', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_SVM_Predictions_vs_Actual.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Residuals Plot
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals (%)', fontsize=11, fontweight='bold')
plt.title('SVM Model Residuals Analysis', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_SVM_Residuals.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Error Distribution
plt.figure(figsize=(10, 6))
abs_errors = np.abs(residuals)
plt.hist(abs_errors, bins=30, color='steelblue', edgecolor='navy', alpha=0.7)
plt.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}%')
plt.xlabel('Absolute Error (%)', fontsize=11, fontweight='bold')
plt.ylabel('Frequency', fontsize=11, fontweight='bold')
plt.title('SVM: Distribution of Absolute Prediction Errors', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_SVM_Error_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature Importance via Permutation
# SVM has no built-in feature importance; permuting each feature and measuring R² drop is the standard approach
print("Calculating feature importance via permutation...")
rng = np.random.default_rng(42)
baseline_r2 = r2_score(y_test.values, svm_model.predict(X_test_scaled))
perm_importance = []
for i in range(len(features)):
    X_permuted = X_test_scaled.copy()
    rng.shuffle(X_permuted[:, i])
    perm_r2 = r2_score(y_test.values, svm_model.predict(X_permuted))
    perm_importance.append(max(0.0, baseline_r2 - perm_r2))

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': perm_importance
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', edgecolor='navy')
plt.xlabel('Importance Score (R² drop on permutation)', fontsize=11, fontweight='bold')
plt.title('SVM: Feature Importance via Permutation', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
             ha='left', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_SVM_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Model Performance Summary
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].text(0.5, 0.7, f'{rmse:.2f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[0].text(0.5, 0.3, 'RMSE (%)', ha='center', va='center', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].axis('off')

axes[1].text(0.5, 0.7, f'{mae:.2f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[1].text(0.5, 0.3, 'MAE (%)', ha='center', va='center', fontsize=12, fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].axis('off')

axes[2].text(0.5, 0.7, f'{r2:.3f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[2].text(0.5, 0.3, 'R² Score', ha='center', va='center', fontsize=12, fontweight='bold')
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].axis('off')

fig.suptitle('SVM Regression Model Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_SVM_Performance_Summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] All visualizations saved to: {output_dir}/")
print(f"[OK] SVM Regression model training complete!")
