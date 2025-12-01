import pandas as pd
from pathlib import Path
import os
import numpy as np

# --- KONFIGURACIJA PUTANJA ---
BASE_PATH = Path('C:/Users/User/Desktop/ISISproj/isis_projekat/')
DATA_PROCESSED_PATH = BASE_PATH / 'data/processed'
DATA_RAW_PATH = BASE_PATH / 'data/raw'

# --- 1. FUNKCIJA ZA OBRADU PRAZNIKA (Preuzeto iz train_model.py) ---
def load_and_clean_holidays(path: Path) -> pd.DataFrame:
    """Učitava i čisti podatke o praznicima, vraćajući DataFrame sa datumima."""
    # Putanja pretpostavlja da je fajl u data/raw/
    try:
        holiday_df = pd.read_excel(path, skiprows=1) 
    except FileNotFoundError:
        print(f"Greška: Fajl praznika nije pronađen na putanji: {path}")
        return pd.DataFrame() # Vraća prazan DF ako fajl ne postoji

    holiday_df.columns = ['Year', 'Day', 'Date', 'Holiday'] 
    holiday_df = holiday_df.drop('Year', axis=1).dropna() 
    
    # Kritično: Konvertujemo datume
    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], errors='coerce')
    holiday_df = holiday_df.dropna(subset=['Date']) 
    
    # Kreiramo set jedinstvenih datuma praznika
    holiday_dates = set(holiday_df['Date'].dt.normalize().dt.date)
    if not holiday_dates:
        print("Upozorenje: Set praznika je prazan nakon čišćenja. Proverite format datuma u Excel fajlu.")
    else:
        print(f"DAL: Uspešno pronađeno {len(holiday_dates)} jedinstvenih datuma praznika.")
        print(f"DAL: Prvi pronađeni praznik: {next(iter(holiday_dates))}")
    return holiday_dates

# --- 2. GLAVNA FUNKCIJA ZA PRIPREMU FINAL_DF ---
def prepare_final_df_with_holidays():
    
    # A. Učitavanje finalnog (nekompletnog) DataFrame-a
    final_df_path = DATA_PROCESSED_PATH / 'final_df.pkl'
    try:
        final_df = pd.read_pickle(final_df_path)
    except FileNotFoundError:
        print(f"Greška: final_df.pkl nije pronađen. Proverite putanju: {final_df_path}")
        return
    
    # B. Učitavanje i obrada praznika
    # Pretpostavljamo da se fajl praznika zove 'holidays.xlsx'
    holiday_file_path = DATA_RAW_PATH / 'US Holidays 2018-2021.xlsx' 
    holiday_dates_set = load_and_clean_holidays(holiday_file_path)

    if not holiday_dates_set:
        print("Upozorenje: Nije pronađen validan set datuma praznika. Nastavljam bez is_holiday kolone.")
        return

    # C. Kreiranje is_holiday kolone
    
    # 1. Osiguravamo da je indeks datuma/vremena (već bi trebalo da bude)
    final_df.index = pd.to_datetime(final_df.index)
    
    # 2. Kreiramo novu kolonu na osnovu poređenja (True/False)
    # Poredimo samo DATUM indeksa sa setom datuma praznika
    #final_df['is_holiday'] = pd.Series(final_df.index.normalize().date).map(lambda x: x in holiday_dates_set)    
    # D. Provera (Opciono, ali preporučeno)
    holiday_dates_str_set = {d.strftime('%Y-%m-%d') for d in holiday_dates_set}

    # Drugo: Konvertujemo indeks DataFrame-a u STRING format (dnevna rezolucija)
    # Koristimo strftime direktno na normalizovanom DatetimeIndexu (najčistije!)
    dates_to_check = final_df.index.normalize().strftime('%Y-%m-%d')

    # Treće: Sada se poređenje vrši između dva seta stringova, garantujući podudarnost
    final_df['is_holiday'] = dates_to_check.map(lambda x: x in holiday_dates_str_set)
    print(f"Originalni DataFrame ima {len(final_df)} redova.")
    print(f"Broj sati koji su praznici: {final_df['is_holiday'].sum()} (Očekujte oko 24 * broj dana praznika).")
    
    # E. SNIMANJE KORIGOVANOG FAJLA
    final_df.to_pickle(final_df_path)
    print(f"\n✅ Uspješno kreiran i SNIMLJEN novi final_df.pkl sa kolonom 'is_holiday' na putanji: {final_df_path}")

# --- POKRETANJE ---
if __name__ == '__main__':
    prepare_final_df_with_holidays()