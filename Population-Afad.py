import pandas as pd
import re
import openpyxl  # Resolves hidden zeros and text formats in Excel

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
son_nufus['MAHALLE'] = son_nufus['MAHALLE'].apply(fix_turkish_letters)

print("\n--- 2. CLEANED POPULATION DATA (FIRST 15 ROWS) ---")
print(son_nufus.head(15))


# <3. FINAL DESCRIPTIVE STATISTICS>

print("\n--- 1. AFAD AREA STATS ---")
afad_stats = temiz_veri[['ALAN_M2', 'KAPASITE']].agg(['min', 'max', 'mean', 'median'])
print(afad_stats.round(2))

print("\n--- 2. NEIGHBORHOOD POPULATION STATS ---")
nufus_stats = son_nufus[['NUFUS']].agg(['min', 'max', 'mean', 'median'])
print(nufus_stats.round(2))

