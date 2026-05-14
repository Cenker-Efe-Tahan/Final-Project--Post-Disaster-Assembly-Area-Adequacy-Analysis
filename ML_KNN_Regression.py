import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
pd.options.mode.chained_assignment = None

output_dir = "ML_KNN-Charts"
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
# FEATURE SCALING (CRITICAL FOR KNN)
print("\nScaling features (essential for KNN distance metrics)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# HYPERPARAMETER TUNING: Find Optimal K
print("\nTuning hyperparameter 'k' via cross-validation...")
k_values = range(3, 31, 2)
cv_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean', weights='distance')
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    print(f"k={k:2d} | CV R² Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Find optimal k
optimal_k = k_values[np.argmax(cv_scores)]
print(f"\n✓ Optimal k: {optimal_k} (R² Score: {max(cv_scores):.4f})")

# ==========================================
# TRAINING with Optimal K
print("\nTraining KNN Regressor with optimal parameters...")
knn_model = KNeighborsRegressor(
    n_neighbors=optimal_k,
    metric='euclidean',
    weights='distance',  # Closer neighbors have more weight
    n_jobs=-1
)
knn_model.fit(X_train_scaled, y_train)

# ==========================================
# PREDICTION & EVALUATION
y_pred = knn_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print()
print("             MODEL PERFORMANCE REPORT             ")
print("==================================================")
print(f"Root Mean Square Error (RMSE) : {rmse:.2f} %")
print(f"Mean Absolute Error (MAE)     : {mae:.2f} %")
print(f"R-Squared (R2) Score          : {r2:.3f}")
print(f"Optimal Neighbors (k)         : {optimal_k}")
print("==================================================\n")

# ==========================================
# VISUALIZATIONS

# 1. K-value Tuning Curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-', linewidth=2, markersize=6, color='steelblue')
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
plt.xlabel('Number of Neighbors (k)', fontsize=11, fontweight='bold')
plt.ylabel('Cross-Validation R² Score', fontsize=11, fontweight='bold')
plt.title('KNN: Hyperparameter Tuning - Optimal k Selection', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_KNN_K_Tuning_Curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Predictions vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.ylabel('Predicted Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.title(f'KNN Predictions vs Actual (R² = {r2:.3f})', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_KNN_Predictions_vs_Actual.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Residuals Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue', edgecolors='navy', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Vulnerability Ratio (%)', fontsize=11, fontweight='bold')
plt.ylabel('Residuals (%)', fontsize=11, fontweight='bold')
plt.title('KNN Model Residuals Analysis', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_KNN_Residuals.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Error Distribution
plt.figure(figsize=(10, 6))
abs_errors = np.abs(residuals)
plt.hist(abs_errors, bins=30, color='steelblue', edgecolor='navy', alpha=0.7)
plt.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}%')
plt.xlabel('Absolute Error (%)', fontsize=11, fontweight='bold')
plt.ylabel('Frequency', fontsize=11, fontweight='bold')
plt.title('KNN: Distribution of Absolute Prediction Errors', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_KNN_Error_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature Importance (Distance-based approximation)
# For KNN, we can approximate importance by feature variance and correlation with target
feature_importance = []
for feature in features:
    correlation = train_df[feature].corr(train_df[target])
    variance = train_df[feature].var()
    importance_score = abs(correlation) * (variance / train_df[features].var().max())
    feature_importance.append(importance_score)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue', edgecolor='navy')
plt.xlabel('Importance Score (Correlation × Variance)', fontsize=11, fontweight='bold')
plt.title('KNN: Feature Importance Approximation', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
             ha='left', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_KNN_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Model Performance Summary
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# RMSE
axes[0].text(0.5, 0.7, f'{rmse:.2f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[0].text(0.5, 0.3, 'RMSE (%)', ha='center', va='center', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)
axes[0].axis('off')

# MAE
axes[1].text(0.5, 0.7, f'{mae:.2f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[1].text(0.5, 0.3, 'MAE (%)', ha='center', va='center', fontsize=12, fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
axes[1].axis('off')

# R²
axes[2].text(0.5, 0.7, f'{r2:.3f}', ha='center', va='center', fontsize=40, fontweight='bold', color='steelblue')
axes[2].text(0.5, 0.3, 'R² Score', ha='center', va='center', fontsize=12, fontweight='bold')
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1)
axes[2].axis('off')

fig.suptitle('KNN Regression Model Performance Metrics', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_KNN_Performance_Summary.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ All visualizations saved to: {output_dir}/")
print(f"✓ KNN Regression model training complete!")
