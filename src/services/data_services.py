import pandas as pd
from datetime import date
from sqlalchemy import text
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# Uvoz iz našeg Sloja za Pristup Podacima (DAL)
# NAPOMENA: Pretpostavljamo da su ove funkcije ispravno definisane u database.db_connector
from database.db_connector import create_db_engine, fetch_training_data, insert_new_input_data, load_ml_resources 

def _sanitize_dataframe(df: pd.DataFrame, required_cols: set) -> pd.DataFrame:
    """
    Sanitizuje imena kolona (mala slova, uklanjanje razmaka) i proverava
    prisutnost obaveznih kolona.
    """
    df = df.copy()
    
    # 1. Sanitizacija: Konvertovanje u mala slova i uklanjanje razmaka
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # 2. Provera i preimenovanje 'datetime' u 'timestamp'
    if 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'timestamp'})
    
    # 3. Provera da li sve obavezne kolone postoje
    available_cols = set(df.columns)
    missing_cols = required_cols - available_cols
        
    if missing_cols:
        missing_cols_str = ', '.join(f"'{col}'" for col in missing_cols)
        available_cols_str = ', '.join(f"'{col}'" for col in available_cols)
        # Ovdje bacamo izuzetak koji će biti uhvaćen u calling funkciji
        raise ValueError(
            f"Greška: Očekivane kolone ({missing_cols_str}) nisu pronađene u učitanom fajlu. "
            f"Dostupne kolone nakon sanitizacije: {available_cols_str}. "
            f"Za ostvarenja, obavezne su 'timestamp' i 'load_kwh'."
        )
        
    return df

# --- 1. FUNKCIJE ZA PRISTUP PODACIMA IZ BAZE ---

def get_training_data(date_from: date, date_to: date) -> pd.DataFrame:
    """
    Servisni sloj poziva DAL za čitanje podataka iz baze za određeni opseg.
    """
    # DAL će proveriti konekciju
    engine = create_db_engine()
    
    # Poziv DAL funkciji koja izvršava SQL upit
    df = fetch_training_data(str(date_from), str(date_to), engine)
    
    if df.empty:
        raise ValueError("Nisu pronađeni podaci za odabrani datumski opseg.")
        
    return df

def insert_new_input_data_to_db(input_df: pd.DataFrame):
    """
    Čuva nezavisne (meteorološke) podatke koji se koriste za prognozu u bazu.
    """
    try:
        # Podesi kolonu 'datetime' u 'timestamp' i ukloni nepotrebne kolone
        df_to_save = input_df.copy()
        
        # PROVERA: Ovdje bi trebalo da se primeni _sanitize_dataframe ako se ulazni podaci čitaju iz fajla
        # Ali, pošto se ova funkcija koristi unutar make_prediction za čuvanje ulaza,
        # pretpostavljamo da su podaci već očišćeni ili se čitaju iz baze.
        if 'datetime' in df_to_save.columns:
            df_to_save = df_to_save.rename(columns={'datetime': 'timestamp'})
        
        # Osigurajte da je timestamp indeks
        df_to_save = df_to_save.set_index(pd.to_datetime(df_to_save['timestamp']))
        
        # Poziv DAL-u
        engine = create_db_engine()
        # insert_new_input_data očekuje DataFrame sa kolonom 'timestamp'
        insert_new_input_data(df_to_save.reset_index(), engine) 
        return "Novi ulazni podaci sačuvani."
    except Exception as e:
        return f"Greška prilikom čuvanja ulaznih podataka: {e}"

def update_actual_results_in_db(actual_data_df: pd.DataFrame) -> str:
    """
    Ažurira tabelu 'prediction_results' stvarnim ostvarenjima (y_actual) 
    i računa greške (MAPE, MAE).
    
    NAPOMENA: actual_data_df MORA da ima kolone 'timestamp' i 'load_kwh'.
    """
    # Zahtevane kolone: timestamp i load_kwh
    required_cols = {'timestamp', 'load_kwh'}
    
    try:
        engine = create_db_engine()
        
        # 1. Sanitizacija stvarnih ostvarenja (Y_actual)
        actual_data_df = _sanitize_dataframe(actual_data_df, required_cols)
        
        # 2. Čitanje postojećih prognoza iz baze za ažuriranje
        query = text("""
            SELECT timestamp, model_name, predicted_load 
            FROM prediction_results 
            WHERE actual_load IS NULL;
        """)
        predictions_df = pd.read_sql(query, engine)
        
        if predictions_df.empty:
            return "Nema prognoza u bazi za ažuriranje stvarnim ostvarenjima (actual_load IS NULL)."

        # 3. Priprema stvarnih ostvarenja (Y_actual)
        actual_data_df['timestamp'] = pd.to_datetime(actual_data_df['timestamp'])
        
        # Ostavljamo samo ključne kolone
        actual_df = actual_data_df[['timestamp', 'load_kwh']].copy()
        
        # Preimenovanje iz 'load_kwh' u 'actual_load' (za usklađivanje sa kolonom u bazi)
        actual_df = actual_df.rename(columns={'load_kwh': 'actual_load'})

        # 4. Spajanje podataka i računanje grešaka
        merged_df = pd.merge(predictions_df, actual_df, on='timestamp', how='inner')
        
        if merged_df.empty:
            return "Ne poklapaju se timestampovi iz prognoze i stvarnih podataka. Proverite datumske opsege."

        results = []
        report_msg = "Rezultati grešaka:\n"
        
        # Računanje grešaka po modelu
        for model_name, group in merged_df.groupby('model_name'):
            # Filtriramo nultu potrošnju za MAPE
            group_for_mape = group[group['actual_load'] > 0].copy()
            
            mape_val = None
            if not group_for_mape.empty:
                 mape_val = mean_absolute_percentage_error(
                     group_for_mape['actual_load'], group_for_mape['predicted_load']
                 ) * 100

            mae_val = mean_absolute_error(
                group['actual_load'], group['predicted_load']
            )
            
            # Dodajemo podatke za ažuriranje
            for index, row in group.iterrows():
                results.append({
                    'timestamp': row['timestamp'],
                    'model_name': model_name,
                    # Ovdje koristimo vrednosti iz grupne kalkulacije greške
                    'actual_load': row['actual_load'], 
                    'mape': mape_val, # MAPE vrednost je ista za celu grupu modela
                    'mae': mae_val, # MAE vrednost je ista za celu grupu modela
                })
                
            # Priprema poruke o uspehu
            mape_display = f"{mape_val:.2f}%" if mape_val is not None else "N/A (Nulta potrošnja)"
            report_msg += f"- {model_name} Model: MAPE={mape_display}, MAE={mae_val:.2f} kWh\n"

        # Pripremamo DataFrame za ažuriranje (jedinstvena kombinacija timestamp/model_name)
        # Ne treba nam drop_duplicates jer se u petlji results pune za sve modele/timestampove
        update_df = pd.DataFrame(results)
        
        # 5. Ažuriranje baze (SQL upit)
        with engine.begin() as connection:
            for index, row in update_df.iterrows():
                # Konverzija MAPE/MAE u None ako je NaN pre upisa u bazu
                mape_db = row['mape'] if pd.notna(row['mape']) else None
                mae_db = row['mae'] if pd.notna(row['mae']) else None
                
                update_query = text("""
                    UPDATE prediction_results
                    SET actual_load = :actual_load, 
                        mape = :mape, 
                        mae = :mae
                    WHERE timestamp = :timestamp
                    AND model_name = :model_name;
                """)
                
                connection.execute(update_query, {
                    'actual_load': row['actual_load'],
                    'mape': mape_db,
                    'mae': mae_db,
                    'timestamp': row['timestamp'].to_pydatetime(), # Konverzija u standardni Python datetime
                    'model_name': row['model_name']
                })
        
        return f"Tabela 'prediction_results' uspešno ažurirana sa stvarnim ostvarenjima i greškama.\n{report_msg}"

    except ValueError as ve:
        # Hvata grešku iz _sanitize_dataframe
        return str(ve) 
    except Exception as e:
        # Hvata sve ostale neočekivane greške
        return f"Došlo je do neočekivane greške pri ažuriranju: {e}"

# --- 2. FUNKCIJE ZA PRISTUP RESURSIMA SA DISKA ---

def get_ml_resources():
    """
    Učitava istrenirane modele i skalere sa diska.
    """
    return load_ml_resources()
