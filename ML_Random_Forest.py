import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
pd.options.mode.chained_assignment = None
output_dir = "ML_Random_Forest-Charts"
os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv("output/Merged_Full_Dataset.csv")
# Recalculate the Target Metric (Vulnerability Ratio)
df['TARGET_AREA'] = df['NUFUS'] * 1.5
df['AREA_DEFICIT'] = (df['TARGET_AREA'] - df['ALAN_M2']).clip(lower=0)
df['VULNERABILITY_RATIO'] = np.where(
    df['TARGET_AREA'] > 0,
    (df['AREA_DEFICIT'] / df['TARGET_AREA']) * 100,
    0
)
df['VULNERABILITY_RATIO'] = df['VULNERABILITY_RATIO'].clip(upper=100)

os.makedirs("Merged_Data_Charts", exist_ok=True)

# Feature Selection
# Features: Environmental Satellite Data
# Target: Urban Risk Score (Vulnerability Ratio)
features = ['NDVI', 'NDBI', 'LST_C']
X = df[features]
y = df['VULNERABILITY_RATIO']
print()
print(f"Total viable neighborhoods for training: {len(df)}")
print("Features utilized: Green Cover (NDVI), Built-up Index (NDBI), Surface Temp (LST_C)")

# Train-Test Split (%70-%30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"Data successfully split: {len(X_train)} Train | {len(X_test)} Test")

# Training
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print()
print("             MODEL PERFORMANCE REPORT             ")
print("==================================================")
print(f"Root Mean Square Error (RMSE) : {rmse:.2f} %")
print(f"R-Squared (R2) Score          : {r2:.3f}")
print("==================================================\n")

# Actual vs Predicted Visual
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred,
    alpha=0.6,
    s=60,
    color='#1f77b4',
    edgecolor='w'
)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Perfect Prediction (Ideal)')
plt.title('Machine Learning Performance: Actual vs. Predicted Risk', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Actual Vulnerability Ratio (%)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Vulnerability Ratio (%)', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/1_ML_Actual_vs_Predicted.png", dpi=300)
plt.close()
print(f"Actual vs Predicted plot saved as '{output_dir}/1_ML_Actual_vs_Predicted.png'")

# Second Visual
feature_importance = rf_model.feature_importances_ * 100
importance_df = pd.DataFrame({
    'Feature': ['Green Cover (NDVI)', 'Built-up Index (NDBI)', 'Surface Temp (LST_C)'],
    'Importance (%)': feature_importance
}).sort_values(by='Importance (%)', ascending=False)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    x='Importance (%)',
    y='Feature',
    data=importance_df,
    hue='Feature',
    legend=False,
    palette='viridis'
)

for p in ax.patches:
    width = p.get_width()
    plt.text(width + 1, p.get_y() + p.get_height()/2. + 0.1, f'{width:.1f}%', ha="left")

plt.xlim(0, 100)
plt.xlabel('Impact Weight on Model Decisions (%)', fontsize=12, fontweight='bold')
plt.ylabel('Environmental Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/2_ML_Feature_Importance.png", dpi=300)
plt.close()
print(f"Feature Importance plot saved as '{output_dir}/2_ML_Feature_Importance.png'")