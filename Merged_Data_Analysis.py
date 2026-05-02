import pandas as pd
import openpyxl
import re
import difflib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None

# ==========================================
# 1. INTELLIGENT MATCH

number_to_word = {
    '1': 'BIR', '2': 'IKI', '3': 'UC', '4': 'DORT', '5': 'BES',
    '6': 'ALTI', '7': 'YEDI', '8': 'SEKIZ', '9': 'DOKUZ', '0': 'SIFIR'
}


def fix_turkish_letters(text):
    if pd.isna(text): return text
    return str(text).upper().translate(str.maketrans("ÇĞİÖŞÜ", "CGIOSU")).strip()


def ultra_clean_key(name):
    if pd.isna(name): return ""
    name = str(name).upper()
    for num, word in number_to_word.items():
        name = name.replace(num, word)

    # Remove full neighborhood suffixes safely (does not affect words like MAHMUDIYE)
    name = name.replace('MAHALLESI', '').replace('MAHALLE', '').replace('MAH.', '')

    # Remove 'MAH' ONLY if it's a standalone word or attached exactly at the end of the string
    name = re.sub(r'\bMAH\b', '', name)
    name = re.sub(r'MAH$', '', name)

    return ''.join(char for char in name if char.isalnum())


def extract_base_name(name):
    if pd.isna(name): return ""
    return re.split(r'[\(\/\-]', str(name))[0].strip()


def intelligent_match(base_name, target_name):
    n_key = ultra_clean_key(base_name)
    a_key = ultra_clean_key(target_name)

    if not n_key or not a_key: return None
    if n_key == a_key: return "EXACT"

    n_base = ultra_clean_key(extract_base_name(base_name))
    a_base = ultra_clean_key(extract_base_name(target_name))

    if n_base == a_base and len(n_base) >= 4:
        parts_n = re.split(r'[\(\/\-]', str(base_name), maxsplit=1)
        parts_a = re.split(r'[\(\/\-]', str(target_name), maxsplit=1)
        n_suffix = ultra_clean_key(parts_n[1]) if len(parts_n) > 1 else ""
        a_suffix = ultra_clean_key(parts_a[1]) if len(parts_a) > 1 else ""

        if n_suffix and a_suffix:
            if (n_suffix in a_suffix) or (a_suffix in n_suffix): return "BASE_EXACT"
            seq_suf = difflib.SequenceMatcher(None, n_suffix, a_suffix)
            if seq_suf.ratio() >= 0.70: return "BASE_EXACT"
        else:
            return "BASE_EXACT"

    if '.' in str(base_name):
        parts = str(base_name).upper().split('.')
        if len(parts) >= 2:
            initial = parts[0].strip()[0:1]
            rest = ultra_clean_key(parts[1])
            if a_key.startswith(initial) and a_key.endswith(rest): return "ABBREVIATION"

    if len(n_key) >= 7 and len(a_key) >= 7:
        if n_key in a_key or a_key in n_key: return "SUBSTRING"

    seq = difflib.SequenceMatcher(None, n_key, a_key)
    if seq.ratio() >= 0.82: return "STRICT_FUZZY"
    return None


# ==========================================
# 2. Read and Clean Part

# A. Area Data
def smart_excel_parser(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        data.append(row)
    df = pd.DataFrame(data)
    df = df[[2, 4, 8]]
    df.columns = ['ILCE', 'MAHALLE', 'ALAN_M2']
    return df


def clean_area(val):
    if pd.isna(val): return 0
    if isinstance(val, str):
        val_str = val.strip().replace('.', '').replace(',', '.')
        try:
            return float(val_str)
        except:
            return 0
    if isinstance(val, (int, float)):
        if (val != int(val)) or (0 < val < 100): return float(val) * 1000
        return float(val)
    return 0


veri_afad = smart_excel_parser("Acil_Toplanma_Alanları.xlsx")
veri_afad['ALAN_M2'] = veri_afad['ALAN_M2'].apply(clean_area)
veri_afad['ILCE'] = veri_afad['ILCE'].apply(fix_turkish_letters)
veri_afad['MAHALLE'] = veri_afad['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')
mahalle_afad = veri_afad.groupby(['ILCE', 'MAHALLE'], as_index=False)['ALAN_M2'].sum()

# B. Population Data
nufus_veri = pd.read_csv("Neighborhood_population.csv", sep="|", skiprows=4)
nufus_veri.columns = ["YIL", "KONUM_METNI", "NUFUS", "BOS"]
temiz_nufus = nufus_veri[['KONUM_METNI', 'NUFUS']].dropna()
temiz_nufus['ILCE'] = temiz_nufus['KONUM_METNI'].apply(
    lambda x: re.search(r'İzmir\((.*?)/', str(x)).group(1).upper() if re.search(r'İzmir\((.*?)/', str(x)) else None)
temiz_nufus['MAHALLE'] = temiz_nufus['KONUM_METNI'].apply(
    lambda x: re.search(r'/([^/]+?)\s*Mah\.?\)', str(x)).group(1).upper() if re.search(r'/([^/]+?)\s*Mah\.?\)',
                                                                                       str(x)) else None)
temiz_nufus['NUFUS'] = temiz_nufus['NUFUS'].astype(int)
son_nufus = temiz_nufus[['ILCE', 'MAHALLE', 'NUFUS']].copy()
son_nufus['ILCE'] = son_nufus['ILCE'].apply(fix_turkish_letters)
son_nufus['MAHALLE'] = son_nufus['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')

# C. (LST ve Sentinel-2)
lst_df = pd.read_csv("izmir_mahalle_lst_final_latest_clean.csv")
s2_df = pd.read_csv("izmir_mahalle_s2_final_latest_clean.csv")
sat_df = pd.merge(lst_df, s2_df, on=['ilce_adi', 'name'], how='inner')
sat_df = sat_df.rename(columns={'ilce_adi': 'ILCE', 'name': 'MAHALLE'})
sat_df['ILCE'] = sat_df['ILCE'].apply(fix_turkish_letters)
sat_df['MAHALLE_CLEAN'] = sat_df['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')

# ==========================================
# 3. Merging with two stages

# Population + Area (Like the old one)
matched_data_1 = []
for ilce in son_nufus['ILCE'].unique():
    nufus_ilce_data = son_nufus[son_nufus['ILCE'] == ilce]
    afad_ilce_data = mahalle_afad[mahalle_afad['ILCE'] == ilce].copy()
    available_afad_indices = afad_ilce_data.index.tolist()

    for _, n_row in nufus_ilce_data.iterrows():
        nufus_mah = n_row['MAHALLE']
        best_match_idx = None

        for a_idx in available_afad_indices:
            afad_mah = afad_ilce_data.loc[a_idx, 'MAHALLE']
            m_type = intelligent_match(nufus_mah, afad_mah)
            if m_type:
                best_match_idx = a_idx
                break

        if best_match_idx is not None:
            matched_afad_row = afad_ilce_data.loc[best_match_idx]
            matched_data_1.append({
                'ILCE': ilce,
                'MAHALLE': nufus_mah,
                'NUFUS': n_row['NUFUS'],
                'ALAN_M2': matched_afad_row['ALAN_M2']
            })
            available_afad_indices.remove(best_match_idx)

birlesik_1 = pd.DataFrame(matched_data_1)
print("\n")
print(f"[-] STAGE 1 (Population + Area Data) Match Count: {len(birlesik_1)} Neighborhoods")

# (Population+Area) + (LST, NDVI, NDBI)
matched_data_final = []
for ilce in birlesik_1['ILCE'].unique():
    b1_ilce_data = birlesik_1[birlesik_1['ILCE'] == ilce]
    sat_ilce_data = sat_df[sat_df['ILCE'] == ilce].copy()
    available_sat_indices = sat_ilce_data.index.tolist()

    for _, b1_row in b1_ilce_data.iterrows():
        nufus_mah = b1_row['MAHALLE']
        best_match_idx = None

        for s_idx in available_sat_indices:
            sat_mah = sat_ilce_data.loc[s_idx, 'MAHALLE_CLEAN']
            m_type = intelligent_match(nufus_mah, sat_mah)
            if m_type:
                best_match_idx = s_idx
                break

        if best_match_idx is not None:
            matched_sat_row = sat_ilce_data.loc[best_match_idx]
            matched_row = b1_row.to_dict()
            matched_row.update({
                'LST_C': matched_sat_row['LST_C'],
                'NDVI': matched_sat_row['NDVI'],
                'NDBI': matched_sat_row['NDBI']
            })
            matched_data_final.append(matched_row)
            available_sat_indices.remove(best_match_idx)

final_df = pd.DataFrame(matched_data_final)
print(f"[-] STAGE 2 (Merging All Data) Match Count: {len(final_df)} Neighborhoods")
print("INFORMATION: The number of neigbourhoods in LST and NDBI-NDVI data is: 1074")
final_df['KISI_BASI_M2'] = (final_df['ALAN_M2'] / final_df['NUFUS']).round(2)

# CSV
os.makedirs("output", exist_ok=True)
output_path = "output/Merged_Full_Dataset.csv"
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"==================================================")
print(f"[SUCCESS] Number of matching neighborhoods in the main data: {len(final_df)}")
print(f"Final dataset saved to: '{output_path}'")
print(f"==================================================")

# ==========================================
# 5. EXACT DETECTION OF MISSING NEIGHBORHOODS

unmatched_sat_neighborhoods = []
# To find the true missing records, simulate the matching loop
# and collect the leftover (unused) indices from the pool.
for ilce in birlesik_1['ILCE'].unique():
    b1_ilce_data = birlesik_1[birlesik_1['ILCE'] == ilce]
    sat_ilce_data = sat_df[sat_df['ILCE'] == ilce].copy()

    available_sat_indices = sat_ilce_data.index.tolist()

    for _, b1_row in b1_ilce_data.iterrows():
        nufus_mah = b1_row['MAHALLE']
        best_match_idx = None

        for s_idx in available_sat_indices:
            sat_mah = sat_ilce_data.loc[s_idx, 'MAHALLE_CLEAN']
            if intelligent_match(nufus_mah, sat_mah):
                best_match_idx = s_idx
                break

        if best_match_idx is not None:
            available_sat_indices.remove(best_match_idx)

    for unused_idx in available_sat_indices:
        unmatched_sat_neighborhoods.append({
            'ILCE': ilce,
            'SATELLITE_NAME': sat_ilce_data.loc[unused_idx, 'MAHALLE']
        })

unmatched_df = pd.DataFrame(unmatched_sat_neighborhoods)

unmatched_report_path = "output/Unmatched_Satellite_Neighborhoods.txt"
with open(unmatched_report_path, "w", encoding="utf-8") as f:
    f.write("*********************************************************************************\n")
    f.write("                 MISSING / UNMATCHED SATELLITE NEIGHBORHOODS REPORT\n")
    f.write("=================================================================================\n")
    f.write(f"Total Unmatched Neighborhoods: {len(unmatched_df)}")

    if not unmatched_df.empty:
        for ilce in sorted(unmatched_df['ILCE'].unique()):
            f.write(f"\n>>> DISTRICT: {ilce}\n")
            ilce_kisi = unmatched_df[unmatched_df['ILCE'] == ilce]
            for _, row in ilce_kisi.iterrows():
                f.write(f"    - {row['SATELLITE_NAME']}\n")
    else:
        f.write("\nPerfect match! No missing neighborhoods found.\n")

print(f"List of exactly {len(unmatched_df)} unmatched neighborhoods saved to: '{unmatched_report_path}'\n")
print(f"*****THE CHARTS ARE BEING CREATED...*****\n")

# ==========================================
# 1. LOAD DATA & CALCULATE DERIVED METRICS

df = pd.read_csv("output/Merged_Full_Dataset.csv")
df['TARGET_AREA'] = df['NUFUS'] * 1.5
df['AREA_DEFICIT'] = (df['TARGET_AREA'] - df['ALAN_M2']).clip(lower=0)
df['VULNERABILITY_RATIO'] = np.where(
    df['TARGET_AREA'] > 0,
    (df['AREA_DEFICIT'] / df['TARGET_AREA']) * 100,
    0
)

df['VULNERABILITY_RATIO'] = df['VULNERABILITY_RATIO'].clip(upper=100)
os.makedirs("Population-Area-Charts", exist_ok=True)

# ==========================================
# 2. CORRELATION HEATMAP

corr_cols = ['KISI_BASI_M2', 'AREA_DEFICIT', 'VULNERABILITY_RATIO', 'LST_C', 'NDVI', 'NDBI']
corr_df = df[corr_cols].copy()
corr_df = corr_df.rename(columns={
    'KISI_BASI_M2': 'Area Per Capita (m²)',
    'AREA_DEFICIT': 'Missing Area (m²)',
    'VULNERABILITY_RATIO': 'Vulnerability Score (%)',
    'LST_C': 'Surface Temp (°C)',
    'NDVI': 'Green Cover (NDVI)',
    'NDBI': 'Built-up Area (NDBI)'
})

corr_matrix = corr_df.corr(method='pearson')

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")

ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": .8},
    annot_kws={"size": 12, "weight": "bold"}
)

plt.title('Urban Vulnerability vs. Environmental Factors', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=11, fontweight='bold')

plt.tight_layout()
os.makedirs("Merged_Data_Charts", exist_ok=True)
plt.savefig("Merged_Data_Charts/1_Correlation_Heatmap.png", dpi=300)
plt.close()
print("Heatmap saved to '1_Correlation_Heatmap.png'")

# ==========================================
sns.set_theme(style="whitegrid")

def create_scatter(data, x_col, y_col, title, x_label, y_label, filename, color):
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=data,
        x=x_col,
        y=y_col,
        scatter_kws={'alpha': 0.5, 's': 50, 'color': color, 'edgecolor': 'w'},
        line_kws={'color': '#d62728', 'linewidth': 3}
    )
    plt.title(title, fontsize=15, fontweight='bold', pad=15)
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"Merged_Data_Charts/{filename}", dpi=300)
    plt.close()

# SCATTER 1: Area Per Capita vs Green Cover (NDVI)
clean_per_capita = df[df['KISI_BASI_M2'] <= 30]
create_scatter(
    clean_per_capita, 'NDVI', 'KISI_BASI_M2',
    'Green Neighborhoods -> Safe Neighborhoods',
    'Green Cover Index (NDVI)', 'Assembly Area Per Capita (m²)',
    '2_Scatter_NDVI_vs_PerCapita.png', '#2ca02c'
)
print("Scatter-1 saved to '2_Scatter_NDVI_vs_PerCapita.png'")

# SCATTER 2: Vulnerability Ratio vs Surface Temperature (LST)
# Only taking the risky ones
at_risk_df = df[df['VULNERABILITY_RATIO'] > 0]

create_scatter(
    at_risk_df, 'VULNERABILITY_RATIO', 'LST_C',
    'Thermal Stress in At-Risk Neighborhoods (>0% Deficit)',
    'Vulnerability Score (Missing Area %)', 'Land Surface Temperature (°C)',
    '3_Scatter_Vulnerability_vs_LST.png', '#ff7f0e'
)
print("Scatter-2 saved as '3_Scatter_Vulnerability_vs_LST.png'")