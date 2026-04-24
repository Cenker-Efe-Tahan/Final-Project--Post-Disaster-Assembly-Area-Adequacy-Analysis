import pandas as pd
import re
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
# Mute unnecessary Pandas warnings
pd.options.mode.chained_assignment = None
import os
os.makedirs("Population-Area-Charts", exist_ok=True)

# Turkish letters
def fix_turkish_letters(text):
    if pd.isna(text): return text
    text = str(text).upper()
    translation_table = str.maketrans("ÇĞİÖŞÜ", "CGIOSU")
    return text.translate(translation_table).strip()


def smart_excel_parser(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active
    data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        data.append(row)

    df = pd.DataFrame(data)
    # Only take  "İlçe, Mahalle and Alan_M2"
    df = df[[2, 4, 8]]
    df.columns = ['ILCE', 'MAHALLE', 'ALAN_M2']
    return df

# ==========================================
# 1. Area Data Preprocessing
veri = smart_excel_parser("Acil_Toplanma_Alanları.xlsx")

def clean_area(val):
    if pd.isna(val): return 0

    if isinstance(val, str):
        val_str = val.strip()
        # (1.000 -> 1000)
        val_str = val_str.replace('.', '')
        # (150,5 -> 150.5)
        val_str = val_str.replace(',', '.')
        try:
            return float(val_str)
        except:
            return 0

    # For excel errors.
    if isinstance(val, (int, float)):
        if (val != int(val)) or (0 < val < 100):
            return float(val) * 1000
        return float(val)

    return 0

veri['ALAN_M2'] = veri['ALAN_M2'].apply(clean_area)
veri['ILCE'] = veri['ILCE'].apply(fix_turkish_letters)
veri['MAHALLE'] = veri['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')

# Sum the all area in the same neighbourhood.
temiz_veri = veri.groupby(['ILCE', 'MAHALLE'], as_index=False)['ALAN_M2'].sum()

print("\n--- SIFIR HATALI TEMİZ VERİ (İLK 15) ---")
print(temiz_veri.head(15).to_string())

# ==========================================
# 2. POPULATION DATA PROCESSING

nufus_veri = pd.read_csv("Neighborhood_population.csv", sep="|", skiprows=4)
nufus_veri.columns = ["YIL", "KONUM_METNI", "NUFUS", "BOS"]
temiz_nufus = nufus_veri[['KONUM_METNI', 'NUFUS']].dropna()


def extract_district(text):
    match = re.search(r'İzmir\((.*?)/', str(text))
    return match.group(1).upper() if match else None

def extract_neighborhood(text):
    match = re.search(r'/([^/]+?)\s*Mah\.?\)', str(text))
    return match.group(1).upper() if match else None

# Applying extractions and standardization
temiz_nufus['ILCE'] = temiz_nufus['KONUM_METNI'].apply(extract_district)
temiz_nufus['MAHALLE'] = temiz_nufus['KONUM_METNI'].apply(extract_neighborhood)
temiz_nufus['NUFUS'] = temiz_nufus['NUFUS'].astype(int)

son_nufus = temiz_nufus[['ILCE', 'MAHALLE', 'NUFUS']]
son_nufus['ILCE'] = son_nufus['ILCE'].apply(fix_turkish_letters)
son_nufus['MAHALLE'] = son_nufus['MAHALLE'].apply(fix_turkish_letters).str.replace(' ', '')

print("\n--- 2. CLEANED POPULATION DATA (FIRST 15 ROWS) ---")
print(son_nufus.head(15))

# ==========================================
# 3. MERGING & PER CAPITA CALCULATION

# Dictionary to map digits to text equivalents
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
    # Cuts off anything after '(', '/', or '-'
    if pd.isna(name): return ""
    return re.split(r'[\(\/\-]', str(name))[0].strip()


def intelligent_match(nufus_name, afad_name):
    n_key = ultra_clean_key(nufus_name)
    a_key = ultra_clean_key(afad_name)

    if not n_key or not a_key: return None

    # RULE 1: Exact Match
    if n_key == a_key:
        return "EXACT"

    # RULE 2: Base Name Match with STRICT SUFFIX CONTROL
    n_base = ultra_clean_key(extract_base_name(nufus_name))
    a_base = ultra_clean_key(extract_base_name(afad_name))
    if n_base == a_base and len(n_base) >= 4:
        # Extract the suffixes (everything after the first delimiter)
        parts_n = re.split(r'[\(\/\-]', str(nufus_name), maxsplit=1)
        parts_a = re.split(r'[\(\/\-]', str(afad_name), maxsplit=1)

        n_suffix = ultra_clean_key(parts_n[1]) if len(parts_n) > 1 else ""
        a_suffix = ultra_clean_key(parts_a[1]) if len(parts_a) > 1 else ""

        # If BOTH have suffixes, they must be compatible!
        if n_suffix and a_suffix:
            # e.g., "GOCBEYLIBUCAGI" and "GOCBEYLI" -> Match!
            if (n_suffix in a_suffix) or (a_suffix in n_suffix):
                return "BASE_EXACT"
            # e.g., "DEREKOY" and "TURANLI" -> REJECT!
            seq_suf = difflib.SequenceMatcher(None, n_suffix, a_suffix)
            if seq_suf.ratio() >= 0.70:
                return "BASE_EXACT"
            # If we reach here, suffixes contradict. Abort RULE 2!
        else:
            # If only one (or neither) has a suffix, it's safe to assume they are the same
            return "BASE_EXACT"

    # RULE 3: Abbreviation Logic (e.g., B.HAYRETTINPASA -> BARBAROSHAYRETTINPASA)
    if '.' in str(nufus_name):
        parts = str(nufus_name).upper().split('.')
        if len(parts) >= 2:
            initial = parts[0].strip()[0:1]
            rest = ultra_clean_key(parts[1])
            if a_key.startswith(initial) and a_key.endswith(rest):
                return "ABBREVIATION"

    # RULE 4: Substring Match
    if len(n_key) >= 7 and len(a_key) >= 7:
        if n_key in a_key or a_key in n_key:
            return "SUBSTRING"

    # RULE 5: Strict Fuzzy Match (Cutoff at 0.82)
    seq = difflib.SequenceMatcher(None, n_key, a_key)
    if seq.ratio() >= 0.82:
        return "STRICT_FUZZY"

    return None


# Group the assembly areas
mahalle_afad = temiz_veri.groupby(['ILCE', 'MAHALLE'])['ALAN_M2'].sum().reset_index()

matched_data = []

for ilce in son_nufus['ILCE'].unique():
    nufus_ilce_data = son_nufus[son_nufus['ILCE'] == ilce]
    afad_ilce_data = mahalle_afad[mahalle_afad['ILCE'] == ilce].copy()

    available_afad_indices = afad_ilce_data.index.tolist()

    for _, n_row in nufus_ilce_data.iterrows():
        nufus_mah = n_row['MAHALLE']
        best_match_idx = None
        match_type = None

        for a_idx in available_afad_indices:
            afad_mah = afad_ilce_data.loc[a_idx, 'MAHALLE']
            m_type = intelligent_match(nufus_mah, afad_mah)

            if m_type:
                best_match_idx = a_idx
                match_type = m_type
                break

        if best_match_idx is not None:
            matched_afad_row = afad_ilce_data.loc[best_match_idx]

            matched_data.append({
                'ILCE': ilce,
                'NUFUS_MAHALLE': nufus_mah,
                'AFAD_MAHALLE': matched_afad_row['MAHALLE'],
                'ESLESME_TURU': match_type,
                'NUFUS': n_row['NUFUS'],
                'ALAN_M2': matched_afad_row['ALAN_M2']
            })

            available_afad_indices.remove(best_match_idx)

# Build the DataFrame
birlesik_veri = pd.DataFrame(matched_data)

birlesik_veri['KISI_BASI_M2'] = (birlesik_veri['ALAN_M2'] / birlesik_veri['NUFUS']).round(2)

print(f"\n--- INTELLIGENT MERGE SUCCESSFUL: {len(birlesik_veri)} Neighborhoods Matched! ---")

# ==========================================
# 3.5. VALIDATION EXPORT & TERMINAL CHECK

birlesik_veri = birlesik_veri.rename(columns={'NUFUS_MAHALLE': 'MAHALLE'})

rapor_df = birlesik_veri[['ILCE', 'MAHALLE', 'AFAD_MAHALLE', 'ESLESME_TURU']]
rapor_df.to_excel("Eslesme_Raporu.xlsx", index=False)
print("[SUCCESS] Validation report saved as 'Eslesme_Raporu.xlsx'.")

smart_matches = birlesik_veri[birlesik_veri['ESLESME_TURU'] != 'EXACT']
print("\n--- SAMPLE OF SMART RULE MATCHES ---")
if not smart_matches.empty:
    print(smart_matches[['ILCE', 'MAHALLE', 'AFAD_MAHALLE', 'ESLESME_TURU']].head(20).to_string())
else:
    print("No smart matches were used. All matches were EXACT.")

# ==========================================
# 4. DESCRIPTIVE STATISTICS & TERMINAL OUTPUT

# Terminal Control for whole data
istatistik = birlesik_veri.groupby('ILCE')['KISI_BASI_M2'].agg(
    Mahalle_Sayisi='count',
    Min_M2='min',
    Medyan_M2='median',
    Ortalama_M2='mean',
    Max_M2='max'
).round(2)

# Sorting from least to most.
istatistik = istatistik.sort_values(by='Ortalama_M2', ascending=True)

pd.set_option('display.max_rows', None)
print("\n" + "="*60)
print("--- PER CAPITA SQUARE METER STATISTICS BY DISTRICT (ALL DATA) ---")
print("="*60)
print(istatistik)


# ==========================================
# 5. DATA VISUALIZATIONS

# Boxplot
plot_veri = birlesik_veri

plt.figure(figsize=(16, 8))
sns.set_theme(style="whitegrid")

ax = sns.boxplot(
    data=plot_veri,
    x='ILCE',
    y='KISI_BASI_M2',
    hue='ILCE',
    palette='viridis',
    legend=False
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylim(0, 30)
plt.ticklabel_format(style='plain', axis='y')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Zoomed Boxplot: Assembly Area Per Capita by District (0-30 m² View)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Area Per Person (m²)', fontsize=12, fontweight='bold')

plt.text(0.5, 0.95, '',
         transform=ax.transAxes, fontsize=10, color='red', fontstyle='italic', ha='center')

plt.tight_layout()
plt.savefig("Population-Area-Charts/1_boxplot.png", dpi=300)
plt.close()
print("[SUCCESS] Zoomed Boxplot saved as '1_boxplot.png'!")


# ZOOMED IN SCATTER PLOT (0-40 m²)
zoomed_veri = birlesik_veri[birlesik_veri['KISI_BASI_M2'] <= 40]

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")

ax = sns.regplot(
    data=zoomed_veri,
    x='NUFUS',
    y='KISI_BASI_M2',
    scatter_kws={'alpha': 0.7, 'color': '#2ca02c', 's': 60, 'edgecolor': 'w'},
    line_kws={'color': '#d62728', 'linewidth': 2.5}
)

plt.ticklabel_format(style='plain', axis='x')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.ylim(0, 40)
plt.title('Zoomed-in View: Population vs Area Per Person (Excluding >40 m² Outliers)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Neighborhood Population', fontsize=12, fontweight='bold')
plt.ylabel('Area Per Person (m²)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Population-Area-Charts/2_scatter.png", dpi=300)
plt.close()
print("[SUCCESS] Zoomed scatter plot saved as '2_scatter.png'!")


# RISK ANALYSIS BAR CHART (< 1.5 m²)
riskli_mahalleler = birlesik_veri[birlesik_veri['KISI_BASI_M2'] < 1.5]
risk_tablosu = riskli_mahalleler.groupby('ILCE').size().reset_index(name='Riskli_Mahalle_Sayisi')
risk_tablosu = risk_tablosu.sort_values(by='Riskli_Mahalle_Sayisi', ascending=False)
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    data=risk_tablosu,
    x='ILCE',
    y='Riskli_Mahalle_Sayisi',
    hue='ILCE',
    palette='Reds_r',
    legend=False
)

for i in ax.containers:
    ax.bar_label(i, padding=3, fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.title('Critical Risk Analysis: Number of Neighborhoods with < 1.5 m² Assembly Area Per Person', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Number of High-Risk Neighborhoods', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Population-Area-Charts/3_bar_risk_analysis.png", dpi=300)
plt.close()
print("[SUCCESS] Risk analysis bar chart saved as '3_bar_risk_analysis.png'!")


# RISK ANALYSIS BAR CHART (RATIO: < 1.5 m²)
toplam_mahalle = birlesik_veri.groupby('ILCE').size().reset_index(name='Toplam_Mahalle')
riskli_sayi = riskli_mahalleler.groupby('ILCE').size().reset_index(name='Riskli_Mahalle_Sayisi')
risk_tablosu = pd.merge(toplam_mahalle, riskli_sayi, on='ILCE', how='left').fillna(0)
risk_tablosu['Riskli_Mahalle_Sayisi'] = risk_tablosu['Riskli_Mahalle_Sayisi'].astype(int)
risk_tablosu['Tehlike_Yuzdesi'] = (risk_tablosu['Riskli_Mahalle_Sayisi'] / risk_tablosu['Toplam_Mahalle']) * 100
risk_tablosu['Etiket'] = risk_tablosu['Riskli_Mahalle_Sayisi'].astype(str) + " / " + risk_tablosu['Toplam_Mahalle'].astype(str)
risk_tablosu = risk_tablosu.sort_values(by='Tehlike_Yuzdesi', ascending=False)
risk_tablosu = risk_tablosu[risk_tablosu['Tehlike_Yuzdesi'] > 0]

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    data=risk_tablosu,
    x='ILCE',
    y='Tehlike_Yuzdesi',
    hue='ILCE',
    palette='Reds_r',
    legend=False
)

for index, p in enumerate(ax.patches):
    x_pos = p.get_x() + p.get_width() / 2
    y_pos = p.get_height()
    etiket = risk_tablosu.iloc[index]['Etiket']
    ax.text(x_pos, y_pos + 1, etiket, ha='center', fontsize=11, fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 110)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=100))
plt.title('Critical Risk Ratio: Percentage of Neighborhoods with < 1.5 m² Per Person', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Percentage of High-Risk Neighborhoods (%)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Population-Area-Charts/4_bar_risk_ratio_analysis.png", dpi=300)
plt.close()
print("[SUCCESS] Risk analysis ratio bar chart saved as '4_bar_risk_ratio_analysis.png'!")

# ==========================================
# POPULATION RISK DISTRIBUTION (STACKED BAR CHART)

birlesik_veri['Risk_Kategorisi'] = birlesik_veri['KISI_BASI_M2'].apply(
    lambda x: 'At-Risk Population (< 1.5 m²)' if x < 1.5 else 'Safe Population (>= 1.5 m²)'
)

# Get Safe and At-Risk population from MATCHED data
matched_nufus = birlesik_veri.groupby(['ILCE', 'Risk_Kategorisi'])['NUFUS'].sum().unstack(fill_value=0)
kategoriler = ['At-Risk Population (< 1.5 m²)', 'Safe Population (>= 1.5 m²)']
matched_nufus = matched_nufus.reindex(columns=kategoriler, fill_value=0)

# Get TOTAL population from the RAW population data (Ground Truth)
raw_nufus = son_nufus.groupby('ILCE')['NUFUS'].sum().rename('Total_Raw_Population')

# Combine them and calculate the "Unanalyzed / Missing" population
final_nufus = pd.concat([raw_nufus, matched_nufus], axis=1).fillna(0)
final_nufus['Unanalyzed Population (No Data)'] = final_nufus['Total_Raw_Population'] - (
    final_nufus['At-Risk Population (< 1.5 m²)'] + final_nufus['Safe Population (>= 1.5 m²)']
)

# Sort districts by Total Raw Population
final_nufus = final_nufus.sort_values(by='Total_Raw_Population', ascending=False)

# Select the 3 columns for plotting
plot_cols = ['At-Risk Population (< 1.5 m²)', 'Safe Population (>= 1.5 m²)', 'Unanalyzed Population (No Data)']
plot_df = final_nufus[plot_cols]

plt.figure(figsize=(16, 8))
sns.set_theme(style="whitegrid")

# Colors: Red (Risk), Green (Safe), Grey (Missing Data)
ax = plot_df.plot(
    kind='bar',
    stacked=True,
    color=['#d62728', '#2ca02c', '#7f7f7f'],
    ax=plt.gca(),
    edgecolor='white',
    width=0.75
)

plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.title('Human Impact: Population Risk Distribution by District (Including Missing Data)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Total Population', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Vulnerability Status', title_fontsize='11', fontsize='10', loc='upper right')
plt.tight_layout()
plt.savefig("Population-Area-Charts/5_population_risk.png", dpi=300)
plt.close()
print("[SUCCESS] Stacked population risk chart saved to '5_population_risk.png'!")

# ==========================================
# FULLY SAFE DISTRICTS DETAILED REPORT (TXT)

# Target fully safe districts based on analysis
target_districts = ['BEYDAG', 'CESME', 'SEFERIHISAR', 'KARABURUN', 'GUZELBAHCE', 'NARLIDERE']

# Filter the matched data for these 6 fully safe districts
safe_matched_df = birlesik_veri[birlesik_veri['ILCE'].isin(target_districts)].copy()

# Filter the raw population data to find missing/unmatched neighborhoods for these districts
raw_target_df = son_nufus[son_nufus['ILCE'].isin(target_districts)].copy()

unmatched_df = pd.merge(raw_target_df, safe_matched_df[['ILCE', 'MAHALLE']], on=['ILCE', 'MAHALLE'], how='left',
                        indicator=True)
unmatched_df = unmatched_df[unmatched_df['_merge'] == 'left_only'][['ILCE', 'MAHALLE', 'NUFUS']]
unmatched_df = unmatched_df.sort_values(by=['ILCE', 'MAHALLE'])

# Sort the matched data for a cleaner report (by district, then per capita area)
safe_matched_df = safe_matched_df.sort_values(by=['ILCE', 'KISI_BASI_M2'], ascending=[True, False])

os.makedirs("output", exist_ok=True)

# Write to TXT file
report_path = "output/fully_safe_districts.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("************************************************************************\n")
    f.write("          FULLY SAFE DISTRICTS DETAILED ANALYSIS REPORT\n")
    f.write("************************************************************************\n\n")

    f.write("* UNMATCHED / MISSING NEIGHBORHOODS\n")
    f.write("The following neighborhoods in these districts could not be matched with Area data.\n")
    f.write("They are excluded from the safety calculations (Treated as No Data).\n\n")

    if not unmatched_df.empty:
        unmatched_report = unmatched_df.copy()
        unmatched_report.columns = ['DISTRICT', 'UNMATCHED_NEIGHBORHOOD', 'POPULATION']
        f.write(unmatched_report.to_string(index=False))
        f.write("\n")
    else:
        f.write("All neighborhoods in these districts were successfully matched.\n\n")

    f.write("------------------------------------------------------------------------\n")
    f.write("DETAILS OF MATCHED SAFE NEIGHBORHOODS\n")
    f.write("------------------------------------------------------------------------\n")

    final_report_df = safe_matched_df[['ILCE', 'MAHALLE', 'NUFUS', 'ALAN_M2', 'KISI_BASI_M2']]
    final_report_df.columns = ['DISTRICT', 'NEIGHBORHOOD', 'POPULATION', 'TOTAL_AREA_M2', 'PER_CAPITA_M2']

    f.write(final_report_df.to_string(index=False))

print(f"[SUCCESS] Fully safe districts detailed report saved to '{report_path}'.")

# ==========================================
# 6. FULLY SAFE DISTRICTS CHART

# Find districts with at least one at-risk neighborhood
at_risk_districts = birlesik_veri[birlesik_veri['KISI_BASI_M2'] < 1.5]['ILCE'].unique()

# Find districts with NO at-risk neighborhoods (our case)
all_districts = son_nufus['ILCE'].unique()
safe_districts = [d for d in all_districts if d not in at_risk_districts]
total_neighborhoods = son_nufus.groupby('ILCE').size().rename('Total_Neighborhoods')

# Get matched (safe) neighborhoods per district
matched_neighborhoods = birlesik_veri.groupby('ILCE').size().rename('Safe_Matched')

completeness_df = pd.concat([total_neighborhoods, matched_neighborhoods], axis=1).fillna(0)
completeness_df['Missing_Unmatched'] = completeness_df['Total_Neighborhoods'] - completeness_df['Safe_Matched']

# Keep only fully safe districts
safe_completeness_df = completeness_df.loc[safe_districts].copy()

# Sort by total neighborhoods descending
safe_completeness_df = safe_completeness_df.sort_values(by='Total_Neighborhoods', ascending=False)

plot_safe_df = safe_completeness_df[['Safe_Matched', 'Missing_Unmatched']]
plot_safe_df.columns = ['Safe & Matched Data', 'Missing / Unmatched Data']

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# (Green: Safe, Yellow: Missing)
ax = plot_safe_df.plot(
    kind='bar',
    stacked=True,
    color=['#2ca02c', '#ffc107'],
    ax=plt.gca(),
    edgecolor='white',
    width=0.65
)

plt.title('Fully Safe Districts: Analyzed Neighborhoods vs. Missing Data', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Number of Neighborhoods', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

for c in ax.containers:
    labels = [f'{int(v.get_height())}' if v.get_height() > 0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontsize=10, fontweight='bold', color='black')

plt.legend(title='Neighborhood Data Status', title_fontsize='11', fontsize='10', loc='upper right')

plt.tight_layout()
plt.savefig("Population-Area-Charts/6_safe_districts_completeness.png", dpi=300)
plt.close()
print("[SUCCESS] Fully safe districts completeness chart saved as '6_safe_districts_completeness.png'!")

# ======================================================================
# MAX VULNERABILITY RATIO: (AT-RISK + UNANALYZED) / TOTAL POPULATION

# Calculate the worst-case scenario risk ratio per district
final_nufus['Max_Vulnerability_Ratio'] = (
    (final_nufus['At-Risk Population (< 1.5 m²)'] + final_nufus['Unanalyzed Population (No Data)'])
    / final_nufus['Total_Raw_Population']
) * 100

risk_ratio_df = final_nufus.reset_index().sort_values(by='Max_Vulnerability_Ratio', ascending=False)

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# (dark red for highest risk)
ax = sns.barplot(
    data=risk_ratio_df,
    x='ILCE',
    y='Max_Vulnerability_Ratio',
    palette='Reds_r',
    hue='ILCE',
    legend=False
)

plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=100))
plt.ylim(0, 105) # Add a little breathing room at the top

for i in ax.containers:
    ax.bar_label(i, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')

plt.title('Worst-Case Scenario: (At-Risk + Missing Data Population) / Total Population', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Maximum Vulnerability Ratio (%)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("Population-Area-Charts/7_max_vulnerability_ratio.png", dpi=300)
plt.close()
print("[SUCCESS] Maximum vulnerability ratio chart saved to '7_max_vulnerability_ratio.png'!")