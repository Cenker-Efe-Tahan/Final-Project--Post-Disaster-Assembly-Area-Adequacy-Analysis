import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

print("==================================================")
df = pd.read_csv("output/Merged_Full_Dataset.csv")

df['TARGET_AREA'] = df['NUFUS'] * 1.5
df['AREA_DEFICIT'] = (df['TARGET_AREA'] - df['ALAN_M2']).clip(lower=0)
df['VULNERABILITY_RATIO'] = np.where(
    df['TARGET_AREA'] > 0,
    (df['AREA_DEFICIT'] / df['TARGET_AREA']) * 100,
    0
)
df['VULNERABILITY_RATIO'] = df['VULNERABILITY_RATIO'].clip(upper=100)

# 70/30 Split
# random_state=42 one time to ensure the physical file is split consistently
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

# Saving the pre-processed and split datasets to physical files
os.makedirs("output", exist_ok=True)
train_df.to_csv("output/Train_Dataset.csv", index=False)
test_df.to_csv("output/Test_Dataset.csv", index=False)

print("[SUCCESS] Data successfully calculated, split, and frozen into physical files!")
print(f"--> TRAIN SET : {len(train_df)} rows saved to 'output/Train_Dataset.csv'")
print(f"--> TEST SET  : {len(test_df)} rows saved to 'output/Test_Dataset.csv'")
print("==================================================")