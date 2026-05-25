import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
pd.options.mode.chained_assignment = None

output_dir = "ML_Random_Forest-Charts"
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_csv("output/Train_Dataset.csv")
test_df = pd.read_csv("output/Test_Dataset.csv")

# Feature Selection - Dropped 'MAHALLE' to prevent overfitting
features = ['ILCE', 'NUFUS', 'LST_C', 'NDVI', 'NDBI']
target = 'ALAN_M2'

# Encode categorical variables for Random Forest
le_ilce = LabelEncoder()

# Fit on combined data to ensure all unique classes are covered
all_ilce = pd.concat([train_df['ILCE'], test_df['ILCE']])
le_ilce.fit(all_ilce)

train_df['ILCE_encoded'] = le_ilce.transform(train_df['ILCE'])
test_df['ILCE_encoded'] = le_ilce.transform(test_df['ILCE'])

features_encoded = ['ILCE_encoded', 'NUFUS', 'LST_C', 'NDVI', 'NDBI']

X_train = train_df[features_encoded]
y_train = train_df[target]
X_test = test_df[features_encoded]
y_test = test_df[target]

print(f"Total viable neighborhoods for training: {len(train_df) + len(test_df)}")
print("Features utilized:", ", ".join(features))
print(f"Loaded Pre-Split Data: {len(X_train)} Train | {len(X_test)} Test")

# ==========================================
# TRAINING
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# ==========================================
# PREDICTION & EVALUATION
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n             MODEL PERFORMANCE REPORT             ")
print("==================================================")
print(f"Root Mean Square Error (RMSE) : {rmse:.2f} m2")
print(f"Mean Absolute Percentage Error(MAPE): {mape:.2%}")
print(f"R-Squared (R2) Score          : {r2:.3f}")
print("==================================================\n")

# ==========================================
# VISUALIZATIONS

# 1. Actual vs Predicted Visual
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
plt.title('Machine Learning Performance: Actual vs. Predicted Area', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Actual Area (m2)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Area (m2)', fontsize=12, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/1_ML_Actual_vs_Predicted.png", dpi=300)
plt.close()
print(f"Actual vs Predicted plot saved as '{output_dir}/1_ML_Actual_vs_Predicted.png'")

# 2. Feature Importance Visual
feature_importance = rf_model.feature_importances_ * 100
importance_df = pd.DataFrame({
    'Feature': features,
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
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/2_ML_Feature_Importance.png", dpi=300)
plt.close()
print(f"Feature Importance plot saved as '{output_dir}/2_ML_Feature_Importance.png'")