import pandas as pd


def fix_turkish_letters(text):
    if pd.isna(text): return text
    return str(text).upper().translate(str.maketrans("ÇĞİÖŞÜ", "CGIOSU")).strip()


def process_odemis(son_nufus):
    df_odemis = pd.read_csv("ODEMIS_TOPLANMA-ALANLARI-LISTESI_101(ÖDEMİŞ).csv", sep=';', encoding='iso-8859-9',
                            skiprows=1, dtype=str)

    df_odemis.columns = [str(c).replace('?', 'I') for c in df_odemis.columns]

    mahalle_col = [c for c in df_odemis.columns if 'MAHALLE ADI' in c][0]
    area_col = [c for c in df_odemis.columns if 'ALANI' in c and 'm2' in c][0]

    kurtarma_sozlugu = {
        'B?RG?': 'BIRGI', '??R?NKOY': 'SIRINKOY', '?LKKUR?UN': 'ILKKURSUN', '?NONU': 'INONU',
        'ALA?ARLI': 'ALASARLI', 'BADEML?': 'BADEMLI', 'BOZDA?': 'BOZDAG', 'BÜLBÜLLER': 'BULBULLER',
        'BÜYÜKAVULCUK': 'BUYUKAVULCUK', 'ÇA?LAYAN': 'CAGLAYAN', 'ÇAMLICA': 'CAMLICA', 'ÇAYIR': 'CAYIR',
        'ÇAYLI': 'CAYLI', 'CEV?ZALAN': 'CEVIZALAN', 'ÇOBANLAR': 'COBANLAR', 'CUMHUR?YET': 'CUMHURIYET',
        'DEM?RC?L?': 'DEMIRCILI', 'DEM?RDERE': 'DEMIRDERE', 'EM?RL?': 'EMIRLI', 'EMM?O?LU': 'EMMIOGLU',
        'ERTU?RUL': 'ERTUGRUL', 'GERCEKL?': 'GERCEKLI', 'GEREL?': 'GERELI', 'GÖLCÜK': 'GOLCUK',
        'GÜNEY': 'GUNEY', 'GÜNLÜCE': 'GUNLUCE', 'HÜRR?YET': 'HURRIYET', 'I?IK': 'ISIK',
        'KARADO?AN': 'KARADOGAN', 'KÖFÜNDERE': 'KOFUNDERE', 'KÖSELER': 'KOSELER',
        'KÜÇÜKAVULCUK': 'KUCUKAVULCUK', 'KÜÇÜKÖREN': 'KUCUKOREN', 'KÜRE': 'KURE', 'KUVVETL?': 'KUVVETLI',
        'M?MARS?NAN': 'MIMARSINAN', 'ME?RUT?YET': 'MESRUTIYET', 'MESC?TL?': 'MESCITLI', 'O?UZLAR': 'OGUZLAR',
        'P?R?NCC?': 'PIRINCCI', 'SEK?KOY': 'SEKIKOY', 'SEYREKL?': 'SEYREKLI', 'SÜLEYMANLAR': 'SULEYMANLAR',
        'TÜRKMEN': 'TURKMEN', 'TÜRKÖNÜ': 'TURKONU', 'ÜÇEYLÜL': 'UCEYLUL', 'ÜÇKONAK': 'UCKONAK',
        'ÜZÜMLÜ': 'UZUMLU', 'VEL?LER': 'VELILER', 'YE??LKOY': 'YESILKOY', 'YEN?CEKOY': 'YENICEKOY',
        'YEN?KOY': 'YENIKOY'
    }

    df_odemis[mahalle_col] = df_odemis[mahalle_col].apply(fix_turkish_letters).str.replace(' ', '')
    df_odemis[mahalle_col] = df_odemis[mahalle_col].replace(kurtarma_sozlugu)

    def parse_area(val):
        if pd.isna(val): return 0
        val = str(val).replace('\n', '').replace('\r', '').strip()
        val = val.replace('.', '')
        try:
            return float(val)
        except:
            return 0

    df_odemis['ALAN_M2'] = df_odemis[area_col].apply(parse_area)

    mahalle_odemis_yeni = df_odemis.groupby(mahalle_col)['ALAN_M2'].sum().reset_index()
    mahalle_odemis_yeni.columns = ['MAHALLE', 'ALAN_M2']

    # Merge with main code
    nufus_odemis = son_nufus[son_nufus['ILCE'] == 'ODEMIS']
    yeni_odemis_birlesik = pd.merge(nufus_odemis, mahalle_odemis_yeni, on='MAHALLE', how='inner')
    yeni_odemis_birlesik['KISI_BASI_M2'] = (yeni_odemis_birlesik['ALAN_M2'] / yeni_odemis_birlesik['NUFUS']).round(2)
    yeni_odemis_birlesik['KAPASITE'] = float('nan')

    return yeni_odemis_birlesik[['ILCE', 'MAHALLE', 'NUFUS', 'ALAN_M2', 'KAPASITE', 'KISI_BASI_M2']]

