from datetime import date
import pandas as pd
from sqlalchemy import DateTime, create_engine, text
from sqlalchemy.engine import Engine
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

# --- KONFIGURACIJA BAZE PODATAKA ---
# AŽURIRAJTE SVOJIM KREDENCIJALIMA
DB_USER = "postgres" 
DB_PASSWORD = "luka24" 
DB_HOST = "localhost"
DB_PORT = "5432" 
DB_NAME = "weatherdb" 
SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- PUTANJE DO FAJLOVA NA DISKU (MODELI I PODACI) ---
# AŽURIRAJTE OVE PUTANJE NA VAŠ SISTEM!
# Bitno: Svi fajlovi koji nisu u bazi (modeli, skaleri, test setovi) ostaju na disku
BASE_PATH = Path('C:/Users/User/Desktop/ISISproj/isis_projekat/') 
DATA_PATH = BASE_PATH / 'data/processed'
MODELS_PATH = BASE_PATH / 'models'

# --- 1. FUNKCIJE ZA POVEZIVANJE SA BAZOM I INICIJALIZACIJA ---

def create_db_engine() -> Engine:
    """Kreira SQLAlchemy engine i testira povezivanje."""
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("DAL: Uspešno povezan sa PostgreSQL bazom.")
        return engine
    except Exception as e:
        print(f"DAL: Greška pri povezivanju sa bazom. Proverite kredencijale: {e}")
        raise

def check_tables_exist(engine: Engine):
    """Proverava da li su ključne tabele kreirane."""
    print("DAL: Očekuje se da su 'training_data', 'new_data_for_prediction', 'prediction_results' tabele kreirane u pgAdminu.")
    # Ovde se može dodati logika za proveru postojanja tabela ako je potrebno

# --- 2. FUNKCIJE ZA UVOZ I ČITANJE PODATAKA IZ BAZE (ZA SERVISNI SLOJ) ---

def insert_training_data(df_path: Path, engine: Engine):
    """
    Učitava final_df.pkl sa diska i upisuje u training_data tabelu.
    Ovo se pokreće samo jednom za inicijalni uvoz.
    """
    try:
        df = pd.read_pickle(df_path)
        
        # A. Mapiranje kolone 'Load' na 'load_kwh'
        if 'Load' not in df.columns:
             df = df.rename(columns={'Load': 'load_kwh'})
        
        # B. Priprema indeksa (koji mora biti 'timestamp')
        # Pretvara indeks (timestamp) u regularnu kolonu:
        df.index.name = 'timestamp' 
        df = df.reset_index() 
        # Osigurava da je tip 'datetime64[ns]'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 3. Upis u training_data tabelu
        df.to_sql(
            name='training_data',
            con=engine,
            if_exists='replace', 
            index=False, # Index je False jer je 'timestamp' sada normalna kolona
            # Koristimo uvezeni SQLAlchemy tip umesto stringa
            dtype={'timestamp': DateTime}
        )
        print(f"DAL: Podaci ({len(df)} redova) uspešno upisani u tabelu 'training_data'.")
    except FileNotFoundError:
        print(f"DAL: Greška: Fajl '{df_path.name}' nije pronađen. Proverite DATA_PATH putanju.")
    except Exception as e:
        print(f"DAL: Greška tokom upisa podataka u bazu: {e}")
        raise

def fetch_training_data(date_from: str, date_to: str, engine: Engine) -> pd.DataFrame:
    """
    Čita podatke iz baze za određeni datumski opseg za potrebe TRENINGA.
    Servisni sloj će pozivati ovu funkciju.
    """
    query = f"""
    SELECT * FROM training_data 
    WHERE timestamp BETWEEN '{date_from}' AND '{date_to}'
    ORDER BY timestamp
    """
    df = pd.read_sql(query, con=engine, index_col='timestamp')
    if 'Load' in df.columns: # Ako se u bazi zove 'load'
        df = df.rename(columns={'Load': 'load_kwh'})
    #print('kolone: ', df.columns)
    return df

def fetch_date_range_from_db(engine: Engine) -> tuple[date, date]:
    """
    Čita najraniji i najkasniji timestamp iz training_data tabele.
    """
    try:
        query = text("SELECT MIN(timestamp), MAX(timestamp) FROM training_data")
        
        with engine.connect() as connection:
            result = connection.execute(query).fetchone()
        
        if result and result[0] and result[1]:
            # Vraća se samo deo datuma (bez vremena) kao Python date objekat
            min_date = result[0].date()
            max_date = result[1].date()
            return min_date, max_date
        else:
            # Vraća default datume ako je tabela prazna
            return date(2020, 1, 1), date(2021, 10, 31)
            
    except Exception as e:
        print(f"DAL: Greška pri čitanju opsega datuma iz baze: {e}")
        # Vraća default datume u slučaju greške konekcije
        return date(2020, 1, 1), date(2021, 10, 31)

# Unutar db_connector.py
def insert_prediction_result(results_df: pd.DataFrame, engine):
    """
    Upisuje NN i RFR prognoze u tabelu prediction_results.
    Prognoze se unose sa NULL vrednostima za actual_load, mape i mae.
    """
    # 1. Priprema NN predikcija
    nn_results = results_df.copy()
    nn_results['model_name'] = 'NN'
    nn_results = nn_results.rename(columns={'NN_Prediction (kWh)': 'predicted_load'})
    nn_results = nn_results.drop(columns=['RFR_Prediction (kWh)'])
    
    # 2. Priprema RFR predikcija
    rfr_results = results_df.copy()
    rfr_results['model_name'] = 'RFR'
    rfr_results = rfr_results.rename(columns={'RFR_Prediction (kWh)': 'predicted_load'})
    rfr_results = rfr_results.drop(columns=['NN_Prediction (kWh)'])

    # Kombinovanje i dodavanje NULL kolona (actual_load, mape, mae, prediction_date)
    final_df = pd.concat([nn_results, rfr_results], ignore_index=True)
    final_df['actual_load'] = None # NULL
    final_df['mape'] = None        # NULL
    final_df['mae'] = None         # NULL
    final_df['prediction_date'] = pd.Timestamp.now()
    
    # Selektovanje kolona prema šemi baze
    final_df = final_df[['timestamp', 'model_name', 'predicted_load', 'actual_load', 'mape', 'mae', 'prediction_date']]
    
    # Unos u bazu
    try:
        final_df.to_sql('prediction_results', engine, if_exists='append', index=False)
        print("Prognoze uspešno unete u prediction_results.")
    except Exception as e:
        print(f"Greška prilikom INSERT-a prognoza: {e}")
        raise


def insert_new_input_data(df: pd.DataFrame, engine):
    """
    INSERT: Upisuje nove meteorološke podatke (X_new) za prognozu u bazu.
    
    :param df: DataFrame sa podacima za period prognoze (bez load_kwh).
    :param engine: Konekcija na bazu.
    """
    try:
        # Tabela je 'new_data_for_prediction'
        df.to_sql('new_data_for_prediction', engine, if_exists='append', index=False)
        print("Novi ulazni podaci (X_new) uspešno sačuvani u bazi.")
        
    except Exception as e:
        print(f"Greška prilikom INSERT-a novih ulaznih podataka: {e}")
        # Možete dodati logovanje greške ovde
        raise

# --- 3. FUNKCIJE ZA UČITAVANJE ML RESURSA (SA DISKA) ---

def load_ml_resources():
    """Učitava istrenirane modele, skalere i test setove sa diska."""
    try:
        model = load_model(MODELS_PATH / 'model.keras')
        rfr_model = joblib.load(MODELS_PATH / 'rfr_model.pkl')
        scaler_X = joblib.load(MODELS_PATH / 'scaler_X.pkl')
        scaler_y = joblib.load(MODELS_PATH / 'scaler_y.pkl')
        X_test_final = joblib.load(MODELS_PATH / 'X_test_final.pkl')
        
        print("DAL: ML resursi su uspešno učitani sa diska.")
        return model, rfr_model, scaler_X, scaler_y, X_test_final
    except Exception as e:
        print(f"DAL: Upozorenje: Greška pri učitavanju ML resursa sa diska. Možda modeli nisu trenirani: {e}")
        return None, None, None, None, None

# --- 4. GLAVNA INICIJALIZACIJA (Uvoz) ---

if __name__ == '__main__':
    # A. Inicijalizacija baze
    db_engine = create_db_engine()
    check_tables_exist(db_engine)
    
    # B. Uvoz final_df.pkl u 'training_data'
    data_file = DATA_PATH / 'final_df.pkl'
    insert_training_data(data_file, db_engine)
    
    # C. Verifikacija
    print("\nVerifikacija uvoza...")
    df_count = pd.read_sql("SELECT COUNT(*) FROM training_data;", con=db_engine).iloc[0, 0]
    print(f"Ukupno redova u training_data: {df_count}")
    
    print("\n--- INICIJALNI UVOZ I DAL SLOJ SPREMNI! ---")