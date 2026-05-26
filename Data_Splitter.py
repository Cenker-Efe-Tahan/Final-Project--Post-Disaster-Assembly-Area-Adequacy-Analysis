import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load the Merged Dataset
df = pd.read_csv("output/Merged_Full_Dataset.csv")

df['TARGET_AREA'] = df['NUFUS'] * 1.5
df['AREA_DEFICIT'] = (df['TARGET_AREA'] - df['ALAN_M2']).clip(lower=0)
df['VULNERABILITY_RATIO'] = np.where(
    df['TARGET_AREA'] > 0,
    (df['AREA_DEFICIT'] / df['TARGET_AREA']) * 100,
    0
)
df['VULNERABILITY_RATIO'] = df['VULNERABILITY_RATIO'].clip(upper=100)

bins = [
    -1,
    1000,
    5000,
    15000,
    50000,
    np.inf
]

labels = [
    '1_Too_Small_Capacity',  # < 1,000 m2
    '2_Small_Capacity',      # 1,000 - 5,000 m2
    '3_Average_Capacity',    # 5,000 - 15,000 m2
    '4_Big_Capacity',        # 15,000 - 50,000 m2
    '5_Too_Big_Capacity'     # > 50,000 m2
]

df['AREA_CATEGORY'] = pd.cut(df['ALAN_M2'], bins=bins, labels=labels)

print("\n--- Seperation in Categories ---")
print(df['AREA_CATEGORY'].value_counts().sort_index())
print("---------------------------------------------------\n")

# Train/Test Split (70% Train, 30% Test)
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)


os.makedirs("output", exist_ok=True)
train_df.to_csv("output/Train_Dataset.csv", index=False)
test_df.to_csv("output/Test_Dataset.csv", index=False)

print(f"--> TRAIN SET : {len(train_df)} neighborhoods")
print(f"--> TEST SET  : {len(test_df)} neighborhoods")
print("==================================================")