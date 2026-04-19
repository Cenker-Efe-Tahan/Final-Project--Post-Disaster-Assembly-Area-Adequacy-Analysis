from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NEIGHBORHOOD_FILE = BASE_DIR / 'Neighborhood_population.csv'
GROWTH_FILE = BASE_DIR / 'ilce_bazli_nufus_degisim_hizi.csv'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_CSV = OUTPUT_DIR / 'mahalle_2026_2027_projeksiyon.csv'
OUTPUT_TXT = OUTPUT_DIR / 'mahalle_2026_2027_projeksiyon_terminal.txt'

TR_LOWER_MAP = str.maketrans({'I': 'ı', 'İ': 'i'})
TR_ASCII_MAP = str.maketrans({
    'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
    'İ': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C',
})


def normalize_key(text: str) -> str:
    text = str(text).translate(TR_LOWER_MAP).lower().strip()
    return text.translate(TR_ASCII_MAP)


def to_ascii_tr(text: str) -> str:
    return str(text).strip().translate(TR_ASCII_MAP)


def parse_neighborhoods(path: Path) -> pd.DataFrame:

    lines = path.read_text(encoding='utf-8-sig').splitlines()
    records = []

    # Dont take the first 5 lines because they are empty.
    for line in lines[5:]:
        parts = line.split('|')
        if len(parts) < 3:
            continue

        location = parts[1].strip()
        pop = parts[2].strip()

        if not location or not pop:
            continue


        inner = location
        if not inner.startswith('İzmir(') or ')-' not in inner:
            continue

        inner = inner[len('İzmir('):]
        inner = inner.rsplit(')-', 1)[0]
        chunks = inner.split('/', 2)
        if len(chunks) != 3:
            continue

        ilce = chunks[0].strip()
        mahalle = chunks[2].strip()
        nufus = int(round(float(pop.replace(',', '.'))))

        records.append({
            'ILCE': ilce,
            'MAHALLE': mahalle,
            'NUFUS_2025': nufus,
        })

    df = pd.DataFrame(records)
    df['join_key'] = df['ILCE'].map(normalize_key)
    return df


def parse_growth(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    records = []

    for line in lines[5:]:
        parts = line.split('|')
        if len(parts) < 3:
            continue

        location = parts[1].strip()
        rate_str = parts[2].strip()

        if not location or not rate_str:
            continue

        if not location.startswith('İzmir(') or ')-' not in location:
            continue

        ilce = location[len('İzmir('):]
        ilce = ilce.rsplit(')-', 1)[0].strip()

        rate = float(rate_str.replace(',', '.'))

        records.append({
            'ILCE': ilce,
            'ARTIS_HIZI_BINDE': rate
        })

    df = pd.DataFrame(records)
    df['join_key'] = df['ILCE'].map(normalize_key)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NEIGHBORHOOD_FILE.exists():
        raise FileNotFoundError(f'Girdi dosyası bulunamadı: {NEIGHBORHOOD_FILE}')
    if not GROWTH_FILE.exists():
        raise FileNotFoundError(f'Girdi dosyası bulunamadı: {GROWTH_FILE}')

    mahalle_df = parse_neighborhoods(NEIGHBORHOOD_FILE)
    artis_df = parse_growth(GROWTH_FILE)

    merged = mahalle_df.merge(artis_df, on='join_key', how='left', suffixes=('', '_ILCE'))

    missing = merged['ARTIS_HIZI_BINDE'].isna().sum()
    if missing:
        raise ValueError(f'{missing} satır için ilçe artış hızı bulunamadı.')

    merged['YILLIK_ARTIS_CARPANI'] = merged['ARTIS_HIZI_BINDE'] / 1000.0

    # For 2026
    merged['TAHMINI_NUFUS_2026'] = (
            merged['NUFUS_2025'] * (1 + merged['YILLIK_ARTIS_CARPANI'])
    ).round().astype(int)

    # For 2027
    merged['TAHMINI_NUFUS_2027'] = (
            merged['NUFUS_2025'] * (1 + merged['YILLIK_ARTIS_CARPANI']) ** 2
    ).round().astype(int)


    result = merged[[
        'ILCE',
        'MAHALLE',
        'NUFUS_2025',
        'ARTIS_HIZI_BINDE',
        'TAHMINI_NUFUS_2026',
        'TAHMINI_NUFUS_2027'
    ]].sort_values(['ILCE', 'MAHALLE']).reset_index(drop=True)

    result['ILCE'] = result['ILCE'].map(to_ascii_tr)
    result['MAHALLE'] = result['MAHALLE'].map(to_ascii_tr)


    result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)

    # Terminal and TXT output
    terminal_output = []
    terminal_output.append('')
    terminal_output.append(result.to_string(index=False))
    text = '\n'.join(terminal_output)

    OUTPUT_TXT.write_text(text, encoding='utf-8')

    print(text)
if __name__ == '__main__':
    main()