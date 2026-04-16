from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NEIGHBORHOOD_FILE = BASE_DIR / 'Neighborhood_population.csv'
GROWTH_FILE = BASE_DIR / 'ilcelere-gore-nufus-artis-hizlari.csv'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_CSV = OUTPUT_DIR / 'mahalle_2yil_projeksiyon.csv'
OUTPUT_TXT = OUTPUT_DIR / 'mahalle_2yil_projeksiyon_terminal.txt'

# Turkish-safe lowercase normalization for joins + ASCII fallback for output
TR_LOWER_MAP = str.maketrans({'I': 'ı', 'İ': 'i'})
TR_ASCII_MAP = str.maketrans({
    'ı': 'i',
    'ğ': 'g',
    'ü': 'u',
    'ş': 's',
    'ö': 'o',
    'ç': 'c',
    'İ': 'I',
    'Ğ': 'G',
    'Ü': 'U',
    'Ş': 'S',
    'Ö': 'O',
    'Ç': 'C',
})


def normalize_key(text: str) -> str:
    text = str(text).translate(TR_LOWER_MAP).lower().strip()
    return text.translate(TR_ASCII_MAP)


def to_ascii_tr(text: str) -> str:
    return str(text).strip().translate(TR_ASCII_MAP)


def parse_neighborhoods(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    records = []

    # Data starts after the first 5 metadata/header lines
    for line in lines[5:]:
        parts = line.split('|')
        if len(parts) < 3:
            continue
        location = parts[1].strip()
        pop = parts[2].strip()
        if not location or not pop:
            continue

        # Example:
        # İzmir(Aliağa/Aliağa Bel./Aşağışakran Mah.)-192999
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
            'MAHALLE_NUFUS_2025': nufus,
        })

    df = pd.DataFrame(records)
    df['join_key'] = df['ILCE'].map(normalize_key)
    return df


def parse_growth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    df['YILLIK_NUFUS_ARTIS_HIZI'] = (
        df['YILLIK_NUFUS_ARTIS_HIZI'].astype(str).str.replace(',', '.', regex=False).astype(float)
    )
    df['join_key'] = df['ILCE'].map(normalize_key)

    # Use the latest available year for each district.
    df = (
        df.sort_values(['join_key', 'TARIH'])
          .groupby('join_key', as_index=False)
          .tail(1)
          .copy()
    )
    return df[['TARIH', 'ILCE', 'NUFUS_TOPLAM', 'YILLIK_NUFUS_ARTIS_HIZI', 'join_key']]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NEIGHBORHOOD_FILE.exists():
        raise FileNotFoundError(f'Girdi dosyası bulunamadı: {NEIGHBORHOOD_FILE}')
    if not GROWTH_FILE.exists():
        raise FileNotFoundError(f'Girdi dosyası bulunamadı: {GROWTH_FILE}')

    mahalle_df = parse_neighborhoods(NEIGHBORHOOD_FILE)
    artis_df = parse_growth(GROWTH_FILE)

    merged = mahalle_df.merge(artis_df, on='join_key', how='left', suffixes=('', '_ILCE'))

    missing = merged['YILLIK_NUFUS_ARTIS_HIZI'].isna().sum()
    if missing:
        raise ValueError(f'{missing} satır için ilçe artış hızı bulunamadı.')

    # User-requested assumption: YILLIK_NUFUS_ARTIS_HIZI is percentage (%).
    # Example: 12.70 -> 0.127 annual rate. 2-year projection is compounded annually.
    merged['YILLIK_ARTIS_ORANI'] = merged['YILLIK_NUFUS_ARTIS_HIZI'] / 100.0
    merged['NUFUS_2_YIL_SONRA'] = (
        merged['MAHALLE_NUFUS_2025'] * (1 + merged['YILLIK_ARTIS_ORANI']) ** 2
    ).round().astype(int)

    result = merged[[
        'ILCE',
        'MAHALLE',
        'MAHALLE_NUFUS_2025',
        'TARIH',
        'YILLIK_NUFUS_ARTIS_HIZI',
        'YILLIK_ARTIS_ORANI',
        'NUFUS_2_YIL_SONRA',
    ]].sort_values(['ILCE', 'MAHALLE']).reset_index(drop=True)

    # Convert displayed/saved text output as requested: ığüşöç -> igusoc
    result['ILCE'] = result['ILCE'].map(to_ascii_tr)
    result['MAHALLE'] = result['MAHALLE'].map(to_ascii_tr)

    result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)

    terminal_output = []
    terminal_output.append('Varsayim: YILLIK_NUFUS_ARTIS_HIZI alani yuzde (%) olarak kabul edildi ve 2025 mahalle nufusu uzerinden 2 yillik bilesik projeksiyon hesaplandi.\n')
    terminal_output.append(f'Toplam mahalle sayisi: {len(result)}')
    terminal_output.append(f'Kullanilan en guncel ilce artis yili: {int(result["TARIH"].max())}')
    terminal_output.append('')
    terminal_output.append(f'Kaydedilen klasor: {to_ascii_tr(OUTPUT_DIR)}')
    terminal_output.append(f'CSV dosyasi: {to_ascii_tr(OUTPUT_CSV)}')
    terminal_output.append(f'Text ciktisi: {to_ascii_tr(OUTPUT_TXT)}')
    terminal_output.append('')
    terminal_output.append(result.to_string(index=False))
    text = '\n'.join(terminal_output)

    OUTPUT_TXT.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
