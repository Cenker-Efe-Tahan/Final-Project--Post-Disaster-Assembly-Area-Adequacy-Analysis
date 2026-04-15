import pandas as pd
import re
import openpyxl  # Resolves hidden zeros and text formats in Excel
import matplotlib.pyplot as plt
import seaborn as sns

# Mute unnecessary Pandas warnings
pd.options.mode.chained_assignment = None


# <TEXT STANDARDIZATION>
def fix_turkish_letters(text):
    # Converts Turkish characters to standard English uppercase
    if pd.isna(text):
        return text
    text = str(text).upper()
    translation_table = str.maketrans("ÇĞİÖŞÜ", "CGIOSU")
    return text.translate(translation_table).strip()


# SMART EXCEL PARSER (THE TERMINATOR VERSION)
def smart_excel_parser(file_path):
    # Bypasses Pandas' blind spots. Fixes hidden zeros, manual text entries, and hidden newlines.
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet = wb.active

    data = []
    # Read starting from row 3 (where actual data begins)
    for row in sheet.iter_rows(min_row=3, values_only=False):
        row_data = []
        for cell in row:
            val = cell.value

            # Target Area (col 8) and Capacity (col 9) columns
            if cell.column in [8, 9]:
                # SCENARIO 1: Excel formatted as number but hides zeros (e.g. 24 -> 24000)
                if isinstance(val, (int, float)):
                    if cell.number_format == '0.000':
                        val = val * 1000

                # SCENARIO 2: Manual text entry with multiple dots AND hidden enter keys
                elif isinstance(val, str):
                    # İŞTE EKSİK OLAN HAYAT KURTARICI SATIR BURASI:
                    val = val.replace('\n', '').replace('\r', '').replace(' ', '')

                    val = val.replace('.', '')  # Remove all thousands separators
                    val = val.replace(',', '.')  # Convert any comma to decimal point
                    try:
                        val = float(val)  # Convert the clean text into a math number
                    except ValueError:
                        pass  # Ignore if it contains weird letters

            row_data.append(val)
        data.append(row_data)

    df = pd.DataFrame(data)
    df.columns = ['SIRA_NO', 'ILCE', 'ALAN_ADI', 'MAHALLE', 'CADDE', 'KAPI_NO', 'KONUM', 'ALAN_M2', 'KAPASITE']
    return df



# <1. AFAD DATA PROCESSING>
veri = smart_excel_parser("afet_toplanma_alanlari.xlsx")

# Keep essential columns including SIRA_NO for verification
temiz_veri = veri[['SIRA_NO', 'ILCE', 'MAHALLE', 'ALAN_M2', 'KAPASITE']].dropna()

# Apply text standardization
temiz_veri['ILCE'] = temiz_veri['ILCE'].apply(fix_turkish_letters)
temiz_veri['MAHALLE'] = temiz_veri['MAHALLE'].apply(fix_turkish_letters)

# Ensure numeric types just in case
temiz_veri['ALAN_M2'] = pd.to_numeric(temiz_veri['ALAN_M2'], errors='coerce')
temiz_veri['KAPASITE'] = pd.to_numeric(temiz_veri['KAPASITE'], errors='coerce')

# --- THE 4 M2 RULE FILTER ---
temiz_veri['RATIO'] = temiz_veri['ALAN_M2'] / temiz_veri['KAPASITE']

# Strict filter: Ratio must be exactly around 4.0 (between 3.9 and 4.1)
temiz_veri = temiz_veri[(temiz_veri['RATIO'] >= 3.9) & (temiz_veri['RATIO'] <= 4.1)]

# Remove impossibly small artifact areas (under 100 m2)
temiz_veri = temiz_veri[temiz_veri['ALAN_M2'] >= 100]

print("\n--- 1. CLEANED AFAD DATA (FIRST 15 ROWS) ---")
print(temiz_veri.head(15))


# <2. POPULATION DATA PROCESSING>

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


# <3. FINAL DESCRIPTIVE STATISTICS>

print("\n--- 1. AFAD AREA STATS ---")
afad_stats = temiz_veri[['ALAN_M2', 'KAPASITE']].agg(['min', 'max', 'mean', 'median'])
print(afad_stats.round(2))

print("\n--- 2. NEIGHBORHOOD POPULATION STATS ---")
nufus_stats = son_nufus[['NUFUS']].agg(['min', 'max', 'mean', 'median'])
print(nufus_stats.round(2))

# MERGING & PER CAPITA CALCULATION
mahalle_afad = temiz_veri.groupby(['ILCE', 'MAHALLE'])[['ALAN_M2', 'KAPASITE']].sum().reset_index()

birlesik_veri = pd.merge(son_nufus, mahalle_afad, on=['ILCE', 'MAHALLE'], how='inner')

birlesik_veri['KISI_BASI_M2'] = (birlesik_veri['ALAN_M2'] / birlesik_veri['NUFUS']).round(2)

print(f"\n--- MERGE SUCCESSFUL: {len(birlesik_veri)} Neighborhoods Matched! ---")


# Terminal Control for whole data
istatistik = birlesik_veri.groupby('ILCE')['KISI_BASI_M2'].agg(
    Mahalle_Sayisi='count',
    Min_M2='min',
    Medyan_M2='median',
    Ortalama_M2='mean',
    Max_M2='max'
).round(2)

istatistik = istatistik.sort_values(by='Ortalama_M2', ascending=True)

pd.set_option('display.max_rows', None)
print("\n" + "="*60)
print("--- PER CAPITA SQUARE METER STATISTICS BY DISTRICT (ALL DATA) ---")
print("="*60)
print(istatistik)


# Boxplot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 8))

ax = sns.boxplot(
    data=birlesik_veri,
    x='ILCE',
    y='KISI_BASI_M2',
    hue='ILCE',
    palette='viridis',
    legend=False
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.ticklabel_format(style='plain', axis='y')
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Distribution of Assembly Area Per Capita by District in İzmir (Including All Outliers)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('District (İlçe)', fontsize=12, fontweight='bold')
plt.ylabel('Area Per Person (m²)', fontsize=12, fontweight='bold')
plt.tight_layout()

plt.savefig("1_boxplot_full_data.png", dpi=300)
plt.close()

print("\n[SUCCESS] Boxplot (with all data) saved as '1_boxplot_full_data.png'!")

# Special control for Selcuk and Torbalı because one of them is too low while other is too high
ozel_ilceler = ['SELCUK', 'TORBALI']
ozel_tablo = birlesik_veri[birlesik_veri['ILCE'].isin(ozel_ilceler)].sort_values(by=['ILCE', 'KISI_BASI_M2'])

print("\n" + "="*60)
print("--- NEIGHBORHOOD LEVEL ASSEMBLY AREA PER CAPITA: SELCUK VS TORBALI ---")
print("="*60)
# We only write the columns we want to see
print(ozel_tablo[['ILCE', 'MAHALLE', 'NUFUS', 'ALAN_M2', 'KISI_BASI_M2']].to_string(index=False))

# Bar for Selcuk and Torbalı to see their neighborhoods speicifcally
# Only Selcuk's and Torbalı's data
karsilastirma_verisi = birlesik_veri[birlesik_veri['ILCE'].isin(['SELCUK', 'TORBALI'])]

plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# We are creating barplot. Y means their neighborhoods, X means their represantative m^2
ax = sns.barplot(
    data=karsilastirma_verisi.sort_values('KISI_BASI_M2', ascending=False),
    y='MAHALLE',
    x='KISI_BASI_M2',
    hue='ILCE',
    palette='Set2'
)


plt.title('Neighborhood Level Assembly Area Per Capita: Selçuk vs Torbalı', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Area Per Person (m²)', fontsize=12, fontweight='bold')
plt.ylabel('Neighborhoods', fontsize=12, fontweight='bold')


plt.xscale('log')
plt.text(0.5, 0.95, '',
         transform=ax.transAxes, fontsize=10, color='red', fontstyle='italic')

plt.tight_layout()
plt.savefig("2_bar_selcuk_torbali.png", dpi=300)
plt.close()

print("\n[SUCCESS] Bar Chart for Selçuk & Torbalı saved as '2_bar_selcuk_torbali.png'!")

def scale_y(val):
    return val if val <= 100 else 100 + (val - 100) * 0.15

birlesik_veri['KISI_BASI_M2_PLOT'] = birlesik_veri['KISI_BASI_M2'].apply(scale_y)

plt.figure(figsize=(12, 8))
ax = sns.regplot(
    data=birlesik_veri, x='NUFUS', y='KISI_BASI_M2_PLOT',
    scatter_kws={'alpha': 0.7, 'color': '#2B5B84', 's': 50, 'edgecolor': 'w'},
    line_kws={'color': '#D9534F', 'linewidth': 2}
)

plt.ticklabel_format(style='plain', axis='x')
plt.gca().xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

max_val = int(birlesik_veri['KISI_BASI_M2'].max())
tick_vals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400]
tick_vals = [v for v in tick_vals if v < max_val]
tick_vals.append(max_val)

plt.yticks([scale_y(v) for v in tick_vals], [str(v) for v in tick_vals])

plt.axhline(100, color='gray', linestyle='--', linewidth=1.5)
plt.text(x=birlesik_veri['NUFUS'].max()*0.5, y=102, s="Scale Compressed Above 100 m²", color='gray', ha='center', fontstyle='italic', fontsize=10)

plt.title('Correlation Between Population and Area Per Person (m²)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Neighborhood Population', fontsize=12, fontweight='bold')
plt.ylabel('Area Per Person (m²)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("3_scatter.png", dpi=300)
plt.close()
print(f"[SUCCESS] Scatter plot saved as '3_scatter.png' (Max Value Plotted: {max_val})")