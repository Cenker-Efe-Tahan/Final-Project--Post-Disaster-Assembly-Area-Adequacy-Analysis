import pandas as pd
import openpyxl
import re
import os
pd.options.mode.chained_assignment = None

def fix_turkish_letters(text):
    if pd.isna(text): return text
    return str(text).upper().translate(str.maketrans("ÇĞİÖŞÜ", "CGIOSU")).strip()

def smart_excel_parser(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    data = []
    for row in sheet.iter_rows(min_row=3, values_only=False):
        row_data = []
        for cell in row:
            val = cell.value
            if cell.column in [8, 9]:
                if isinstance(val, (int, float)):
                    if cell.number_format == '0.000': val = val * 1000
                elif isinstance(val, str):
                    val = val.replace('\n', '').replace('\r', '').replace(' ', '')
                    val = val.replace('.', '').replace(',', '.')
                    try:
                        val = float(val)
                    except ValueError:
                        pass
            row_data.append(val)
        data.append(row_data)
    df = pd.DataFrame(data)
    df.columns = ['SIRA_NO', 'ILCE', 'ALAN_ADI', 'MAHALLE', 'CADDE', 'KAPI_NO', 'KONUM', 'ALAN_M2', 'KAPASITE']
    return df

# Reading AFAD data.
veri = smart_excel_parser("afet_toplanma_alanlari.xlsx")
temiz_veri = veri[['SIRA_NO', 'ILCE', 'MAHALLE', 'ALAN_M2', 'KAPASITE']].dropna()
temiz_veri['ILCE'] = temiz_veri['ILCE'].apply(fix_turkish_letters)
temiz_veri['MAHALLE'] = temiz_veri['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')
temiz_veri['ALAN_M2'] = pd.to_numeric(temiz_veri['ALAN_M2'], errors='coerce')
temiz_veri['KAPASITE'] = pd.to_numeric(temiz_veri['KAPASITE'], errors='coerce')

# 4 square meters rule
temiz_veri['RATIO'] = temiz_veri['ALAN_M2'] / temiz_veri['KAPASITE']
temiz_veri = temiz_veri[(temiz_veri['RATIO'] >= 3.9) & (temiz_veri['RATIO'] <= 4.1)]
temiz_veri = temiz_veri[temiz_veri['ALAN_M2'] >= 100]

mahalle_afad = temiz_veri.groupby(['ILCE', 'MAHALLE'])['ALAN_M2'].sum().reset_index()

# Reading population data.
proj_df = pd.read_csv("output/neighbourhood_2026_2027_projection.csv")
proj_df['ILCE'] = proj_df['ILCE'].apply(fix_turkish_letters)
proj_df['MAHALLE'] = proj_df['MAHALLE'].apply(fix_turkish_letters).str.replace('MAH.', '').str.replace(' ', '')

# Merge part.
merged_df = pd.merge(proj_df, mahalle_afad, on=['ILCE', 'MAHALLE'], how='inner')

merged_df['KISI_BASI_M2_2025'] = (merged_df['ALAN_M2'] / merged_df['NUFUS_2025']).round(2)
merged_df['KISI_BASI_M2_2026'] = (merged_df['ALAN_M2'] / merged_df['TAHMINI_NUFUS_2026']).round(2)
merged_df['KISI_BASI_M2_2027'] = (merged_df['ALAN_M2'] / merged_df['TAHMINI_NUFUS_2027']).round(2)

# Risk analysis (under 1.5 m^2)
# Getting all neighborhoods that fall below 1.5 in ANY of the years 2025, 2026, or 2027.
# This ensures neighborhoods like Vatan, which later exit the risk status, are still included in the final table.
risk_df = merged_df[(merged_df['KISI_BASI_M2_2025'] < 1.5) |
                    (merged_df['KISI_BASI_M2_2026'] < 1.5) |
                    (merged_df['KISI_BASI_M2_2027'] < 1.5)].copy()

# Sort from narrowest to widest area.
risk_df = risk_df.sort_values(by='KISI_BASI_M2_2027', ascending=True)

final_rapor = risk_df[['ILCE', 'MAHALLE', 'ALAN_M2', 'NUFUS_2025', 'TAHMINI_NUFUS_2026', 'TAHMINI_NUFUS_2027',
                       'KISI_BASI_M2_2025', 'KISI_BASI_M2_2026', 'KISI_BASI_M2_2027']]

# Neighborhoods entering risk status in 2027 due to population growth (Safe in 2025)
entering_risk = merged_df[(merged_df['KISI_BASI_M2_2025'] >= 1.5) & (merged_df['KISI_BASI_M2_2027'] < 1.5)]

# Neighborhoods exiting risk status in 2027 due to population decline (At-risk in 2025)
exiting_risk = merged_df[(merged_df['KISI_BASI_M2_2025'] < 1.5) & (merged_df['KISI_BASI_M2_2027'] >= 1.5)]

# Exporting
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

print(f"Report saved to 'output/Future_Risk_Projection.txt'.")