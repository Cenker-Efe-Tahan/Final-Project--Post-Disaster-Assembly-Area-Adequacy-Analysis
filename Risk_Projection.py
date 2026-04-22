import pandas as pd
import openpyxl
import re
import os
import difflib

pd.options.mode.chained_assignment = None

# ==========================================
# 1. TEXT, AREA STANDARDIZATION & MATCH ENGINE

def fix_turkish_letters(text):
    if pd.isna(text): return text
    return str(text).upper().translate(str.maketrans("ÇĞİÖŞÜ", "CGIOSU")).strip()


number_to_word = {
    '1': 'BIR', '2': 'IKI', '3': 'UC', '4': 'DORT', '5': 'BES',
    '6': 'ALTI', '7': 'YEDI', '8': 'SEKIZ', '9': 'DOKUZ', '0': 'SIFIR'
}


def ultra_clean_key(name):
    if pd.isna(name): return ""
    name = str(name).upper()
    for num, word in number_to_word.items():
        name = name.replace(num, word)
    name = name.replace('MAHALLESI', '').replace('MAH.', '').replace('MAH', '')
    return ''.join(char for char in name if char.isalnum())


def extract_base_name(name):
    if pd.isna(name): return ""
    return re.split(r'[\(\/\-]', str(name))[0].strip()


def intelligent_match(nufus_name, area_name):
    n_key = ultra_clean_key(nufus_name)
    a_key = ultra_clean_key(area_name)
    if not n_key or not a_key: return None
    if n_key == a_key: return "EXACT"

    n_base = ultra_clean_key(extract_base_name(nufus_name))
    a_base = ultra_clean_key(extract_base_name(area_name))
    if n_base == a_base and len(n_base) >= 4:
        parts_n = re.split(r'[\(\/\-]', str(nufus_name), maxsplit=1)
        parts_a = re.split(r'[\(\/\-]', str(area_name), maxsplit=1)
        n_suffix = ultra_clean_key(parts_n[1]) if len(parts_n) > 1 else ""
        a_suffix = ultra_clean_key(parts_a[1]) if len(parts_a) > 1 else ""
        if n_suffix and a_suffix:
            if (n_suffix in a_suffix) or (a_suffix in n_suffix): return "BASE_EXACT"
            seq_suf = difflib.SequenceMatcher(None, n_suffix, a_suffix)
            if seq_suf.ratio() >= 0.70: return "BASE_EXACT"
        else:
            return "BASE_EXACT"

    if '.' in str(nufus_name):
        parts = str(nufus_name).upper().split('.')
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
# 2. EXCEL PARSER & DATA PREPARATION

def smart_excel_parser(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    data = []
    # Read starting from row 2
    for row in sheet.iter_rows(min_row=2, values_only=True):
        data.append(row)
    df = pd.DataFrame(data)
    # Using columns 2, 4, 8 for District, Neighborhood, Area in the new dataset
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


# (New Dataset)
veri = smart_excel_parser("Acil_Toplanma_Alanları.xlsx")
veri['ALAN_M2'] = veri['ALAN_M2'].apply(clean_area)
veri['ILCE'] = veri['ILCE'].apply(fix_turkish_letters)
veri['MAHALLE'] = veri['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')
mahalle_area = veri.groupby(['ILCE', 'MAHALLE'])['ALAN_M2'].sum().reset_index()

# Reading population projection data
proj_df = pd.read_csv("output/neighbourhood_2026_2027_projection.csv")
proj_df['ILCE'] = proj_df['ILCE'].apply(fix_turkish_letters)
proj_df['MAHALLE'] = proj_df['MAHALLE'].apply(fix_turkish_letters).str.replace('MAH.', '').str.replace(' ', '')

# Merge Part
matched_data = []

for ilce in proj_df['ILCE'].unique():
    nufus_ilce_data = proj_df[proj_df['ILCE'] == ilce]
    area_ilce_data = mahalle_area[mahalle_area['ILCE'] == ilce].copy()

    available_area_indices = area_ilce_data.index.tolist()

    for _, n_row in nufus_ilce_data.iterrows():
        nufus_mah = n_row['MAHALLE']
        best_match_idx = None

        for a_idx in available_area_indices:
            area_mah = area_ilce_data.loc[a_idx, 'MAHALLE']
            if intelligent_match(nufus_mah, area_mah):
                best_match_idx = a_idx
                break

        if best_match_idx is not None:
            matched_area_row = area_ilce_data.loc[best_match_idx]

            matched_data.append({
                'ILCE': ilce,
                'MAHALLE': nufus_mah,
                'ALAN_M2': matched_area_row['ALAN_M2'],
                'NUFUS_2025': n_row['NUFUS_2025'],
                'TAHMINI_NUFUS_2026': n_row['TAHMINI_NUFUS_2026'],
                'TAHMINI_NUFUS_2027': n_row['TAHMINI_NUFUS_2027']
            })
            available_area_indices.remove(best_match_idx)

merged_df = pd.DataFrame(matched_data)

# ==========================================
# 4. METRICS & PROJECTION CALCULATIONS

merged_df['KISI_BASI_M2_2025'] = (merged_df['ALAN_M2'] / merged_df['NUFUS_2025']).round(2)
merged_df['KISI_BASI_M2_2026'] = (merged_df['ALAN_M2'] / merged_df['TAHMINI_NUFUS_2026']).round(2)
merged_df['KISI_BASI_M2_2027'] = (merged_df['ALAN_M2'] / merged_df['TAHMINI_NUFUS_2027']).round(2)

# Risk analysis (under 1.5 m^2)
# Getting all neighborhoods that fall below 1.5 in ANY of the years 2025, 2026, or 2027.
risk_df = merged_df[(merged_df['KISI_BASI_M2_2025'] < 1.5) |
                    (merged_df['KISI_BASI_M2_2026'] < 1.5) |
                    (merged_df['KISI_BASI_M2_2027'] < 1.5)].copy()

# Sort from narrowest to widest area based on 2027
risk_df = risk_df.sort_values(by='KISI_BASI_M2_2027', ascending=True)

final_rapor = risk_df[['ILCE', 'MAHALLE', 'ALAN_M2', 'NUFUS_2025', 'TAHMINI_NUFUS_2026', 'TAHMINI_NUFUS_2027',
                       'KISI_BASI_M2_2025', 'KISI_BASI_M2_2026', 'KISI_BASI_M2_2027']]

# Neighborhoods entering risk status in 2027 due to population growth (Safe in 2025)
entering_risk = merged_df[(merged_df['KISI_BASI_M2_2025'] >= 1.5) & (merged_df['KISI_BASI_M2_2027'] < 1.5)]

# Neighborhoods exiting risk status in 2027 due to population decline (At-risk in 2025)
exiting_risk = merged_df[(merged_df['KISI_BASI_M2_2025'] < 1.5) & (merged_df['KISI_BASI_M2_2027'] >= 1.5)]

# ==========================================
# 5. EXPORTING REPORTS

os.makedirs("output", exist_ok=True)

csv_path = "output/Future_Risk_Projection.csv"
final_rapor.to_csv(csv_path, index=False, encoding='utf-8-sig')

txt_path = "output/Future_Risk_Projection.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("========================================================================\n")
    f.write("      2025 - 2027 IZMIR NEIGHBORHOODS ANNUAL DISASTER RISK PROJECTION    \n")
    f.write("========================================================================\n")

    risk25 = len(merged_df[merged_df['KISI_BASI_M2_2025'] < 1.5])
    risk26 = len(merged_df[merged_df['KISI_BASI_M2_2026'] < 1.5])
    risk27 = len(merged_df[merged_df['KISI_BASI_M2_2027'] < 1.5])

    f.write(f"Total Number of At-Risk Neighborhoods (2025): {risk25}\n")
    f.write(f"Total Number of At-Risk Neighborhoods (2026): {risk26}\n")
    f.write(f"Total Number of At-Risk Neighborhoods (2027): {risk27}\n\n")

    f.write("Explanation of Why the Number Remains Stable?\n")
    f.write(
        "Small districts of Izmir (Torbali, Cigli, etc.) are receiving migration, causing new neighborhoods to fall into the at-risk category due to population growth. "
        "Oppositely, central districts (Karabaglar, Bayrakli, etc.) are experiencing outward migration, leading to population decline. "
        "This dynamic creates a circulation between entries and exits in the at-risk list and this is balancing the overall total.\n\n")

    f.write(">>> NEIGHBORHOODS ENTERING 'AT-RISK' STATUS DUE TO POPULATION GROWTH <<<\n")
    if len(entering_risk) > 0:
        for _, row in entering_risk.iterrows():
            f.write(
                f"-> {row['ILCE']} - {row['MAHALLE']}: (2025: {row['KISI_BASI_M2_2025']} m² --> 2027: {row['KISI_BASI_M2_2027']} m²) "
                "[Now entered the dangerous neighborhood status]\n")
    else:
        f.write("- None\n")

    f.write("\n>>> NEIGHBORHOODS EXITING RISK DUE TO POPULATION DECLINE <<<\n")
    if len(exiting_risk) > 0:
        for _, row in exiting_risk.iterrows():
            f.write(
                f"-> {row['ILCE']} - {row['MAHALLE']}: (2025: {row['KISI_BASI_M2_2025']} m² --> 2027: {row['KISI_BASI_M2_2027']} m²) "
                "[Exited the dangerous neighborhoods category]\n")
    else:
        f.write("- None\n")

    f.write("\n------------------------------------------------------------------------\n")
    f.write("LIST OF ALL AT-RISK AND RECOVERED NEIGHBORHOODS ACROSS ALL YEARS\n")
    f.write("------------------------------------------------------------------------\n")
    f.write(final_rapor.to_string(index=False))

print(f"[SUCCESS] Projection complete. Reports saved to 'output/Future_Risk_Projection.txt' and CSV.")