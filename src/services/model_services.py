# src/services/model_services.py

import pandas as pd
import numpy as np
from datetime import date
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Uvoz funkcija za pristup podacima (Service-to-Service poziv)
from services.data_services import get_training_data, get_ml_resources
# Uvoz iz DAL-a (za čuvanje/insert u bazu i putanje)
from database.db_connector import MODELS_PATH, insert_prediction_result, create_db_engine
COLS_TO_SCALE = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                           'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                           'solarradiation']

# --- POMOĆNA FUNKCIJA (Obavezna za ciklično kodiranje) ---
def encode_cyclic_feature(df, col, max_val):
    """Ciklično kodiranje vremenskih karakteristika (Sat, Dan, Mesec)."""
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    df = df.drop(col, axis=1)
    return df


# --- 1. FUNKCIJA ZA TRENING MODELA (Ispunjava zahtev TRENING PODATAKA) ---

def train_and_save_models(date_from: date, date_to: date):
    """
    Pokreće trening modela na podacima izabranim iz baze.
    """
    # 1. Čitanje podataka iz baze (Poziv Data Services)
    df = get_training_data(date_from, date_to)
    
    # 2. FEATURE ENGINEERING 
    EXCLUDE_COLS = ['conditions'] 
    
    cols_to_convert = [col for col in df.columns if col not in EXCLUDE_COLS and col != df.index.name]
    
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    df = df.dropna(subset=cols_to_convert)
    
    # Kreiranje vremenskih karakteristika
    df.index = pd.to_datetime(df.index)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear

    # Ciklično kodiranje
    df = encode_cyclic_feature(df, 'hour', 24)
    df = encode_cyclic_feature(df, 'day_of_week', 7)
    df = encode_cyclic_feature(df, 'month', 12)
    df = encode_cyclic_feature(df, 'day_of_year', 366)
    
    # One-Hot Encoding za 'conditions'
    df = pd.get_dummies(df, columns=['conditions'], prefix='cond', drop_first=True)
    
    # 3. Finalizacija kolona
    X = df.drop(['load_kwh', 'solarenergy', 'uvindex'], axis=1) 
    y = df['load_kwh']
    
    # 4. Podela na trening i test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # 5. KREIRAJTE NE-SKALIRANE KOPIJE ZA RFR
    X_train_unscaled = X_train.copy().astype('float32')
    X_test_unscaled = X_test.copy().astype('float32')
    
    # 6. SKALIRANJE SAMO ZA NN
    numerical_cols_to_scale = COLS_TO_SCALE
    
    # 6.1. Izdvajanje podataka za skaliranje
    X_train_to_scale = X_train[numerical_cols_to_scale].astype('float32')
    X_test_to_scale = X_test[numerical_cols_to_scale].astype('float32')
    
    # Izdvajanje neskaliranih kolona
    X_train_unscaled_part = X_train.drop(columns=numerical_cols_to_scale).astype('float32')
    X_test_unscaled_part = X_test.drop(columns=numerical_cols_to_scale).astype('float32')
    
    scaler_X = StandardScaler()
    
    # 6.2. Skaliranje isključivo kontinuiranih podataka
    X_train_scaled = scaler_X.fit_transform(X_train_to_scale)
    X_test_scaled = scaler_X.transform(X_test_to_scale)
    
    # 6.3. Kreiranje finalnih skaliranih skupova za NN
    X_train_nn = pd.concat([
        pd.DataFrame(X_train_scaled, columns=numerical_cols_to_scale, index=X_train_to_scale.index),
        X_train_unscaled_part
    ], axis=1)

    X_test_nn = pd.concat([
        pd.DataFrame(X_test_scaled, columns=numerical_cols_to_scale, index=X_test_to_scale.index),
        X_test_unscaled_part
    ], axis=1)
    
    # 6.4. Usklađivanje redosleda kolona
    X_train_nn = X_train_nn[X.columns].astype('float32')
    X_test_nn = X_test_nn[X.columns].astype('float32')
    
    # 7. SKALIRANJE Y ZA NN
     # 7. SKALIRANJE Y - POTPUNO DRUGAČIJI PRISTUP
    #print('\n--- SKALIRANJE Y VARIJABLE ---')
    
    # Opcija 1: NEMA SKALIRANJA Y - najjednostavnije rješenje
    y_train_nn = y_train.values.reshape(-1, 1).astype('float32')
    y_test_nn = y_test.values.reshape(-1, 1).astype('float32')
    
    # Opcija 2: Blago skaliranje ako je potrebno
    # scaler_y = StandardScaler()
    # y_train_nn = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    # y_test_nn = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    #print(f'y_train_nn - min: {y_train_nn.min():.2f}, max: {y_train_nn.max():.2f}, std: {y_train_nn.std():.2f}')
    
    # 8. PROMJENA ARHITEKTURE - DODAJTE VIŠE SLOJEVA I NEURONA
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_nn.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # LINEARNA AKTIVACIJA ZA REGRESIJU
    ])
    
    # 9. AGRESIVNIJI OPTIMIZER I LEARNING RATE
    model.compile(
        optimizer=Adam(learning_rate=0.01),  # POVEĆAN LEARNING RATE
        loss='mse',  # Vratite se na MSE
        metrics=['mae']
    )
    
    # 10. SMANJITE EARLY STOPPING PATIENCE
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=8,  # SMANJEN PATIENCE
        min_delta=100,  # VECI MIN_DELTA za original scale
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=0.0001,
        verbose=1
    )
    
    #print('\n--- POČETAK TRENINGA NN ---')
    print(f'Input shape: {X_train_nn.shape}')
    print(f'Output range: {y_train_nn.min():.2f} - {y_train_nn.max():.2f}')
    
    history = model.fit(
        X_train_nn, y_train_nn,  # KORISTITE NE-SKALIRANE Y VRIJEDNOSTI
        epochs=100,
        batch_size=32,
        validation_data=(X_test_nn, y_test_nn),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 11. PREDIKCIJA BEZ SKALIRANJA/DE-SKALIRANJA
    nn_predictions = model.predict(X_test_nn).flatten()
    
    # 12. DETALJNA DIJAGNOSTIKA
    #print('\n--- DUBLJA ANALIZA ---')
    print(f'NN predikcije raw - min: {nn_predictions.min():.2f}, max: {nn_predictions.max():.2f}')
    print(f'NN predikcije stats - mean: {nn_predictions.mean():.2f}, std: {nn_predictions.std():.2f}')
    
    # Provjerite da li model uopšte uči pattern
    sample_predictions = model.predict(X_train_nn[:100]).flatten()
    #print(f'Uzorak train predikcija - min: {sample_predictions.min():.2f}, max: {sample_predictions.max():.2f}')
    
    # 13. RFR TRENING
    #print('\n--- TRENING RFR MODELA ---')
    rfr_model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1, 
        max_depth=15
    )
    rfr_model.fit(X_train_unscaled, y_train)
    rfr_predictions = rfr_model.predict(X_test_unscaled)
    
    # 14. FINALNA DIJAGNOSTIKA
    #print('\n--- DIJAGNOSTIKA PREDIKCIJA ---')
    #print(f'Stvarne vrednosti:     min={y_test.min():.2f}, max={y_test.max():.2f}, std={y_test.std():.2f}')
    print(f'NN predikcije:         min={nn_predictions.min():.2f}, max={nn_predictions.max():.2f}, std={nn_predictions.std():.2f}')
    print(f'RFR predikcije:        min={rfr_predictions.min():.2f}, max={rfr_predictions.max():.2f}, std={rfr_predictions.std():.2f}')
    
    # 15. METRIKE
    nn_mape = mean_absolute_percentage_error(y_test, nn_predictions) * 100
    rfr_mape = mean_absolute_percentage_error(y_test, rfr_predictions) * 100
    nn_mae = mean_absolute_error(y_test, nn_predictions)
    rfr_mae = mean_absolute_error(y_test, rfr_predictions)
    
    #print(f'\n--- REZULTATI ---')
    print(f'NN  - MAE: {nn_mae:.2f}, MAPE: {nn_mape:.2f}%')
    print(f'RFR - MAE: {rfr_mae:.2f}, MAPE: {rfr_mape:.2f}%')
    
    # 16. ČUVANJE MODELA BEZ SCALER_Y
    joblib.dump(scaler_X, MODELS_PATH / 'scaler_X.pkl')
    # joblib.dump(scaler_y, MODELS_PATH / 'scaler_y.pkl')  # NE ČUVAMO SCALER_Y
    model.save(MODELS_PATH / 'model.keras')
    joblib.dump(rfr_model, MODELS_PATH / 'rfr_model.pkl')
    joblib.dump(X_test_unscaled, MODELS_PATH / 'X_test_final.pkl')
    joblib.dump(X_train_nn.columns.tolist(), MODELS_PATH / 'feature_columns.pkl')
    
    return {
        "status": f"Trening uspešno izvršen! NN MAPE: {nn_mape:.2f}%, RFR MAPE: {rfr_mape:.2f}%",
        "nn_mape": nn_mape,
        "rfr_mape": rfr_mape,
        "nn_mae": nn_mae,
        "rfr_mae": rfr_mae,
        "y_test": y_test.values,
        "nn_predictions": nn_predictions,
        "rfr_predictions": rfr_predictions
    }


# --- 2. FUNKCIJA ZA PROGNOZU (Ispunjava zahtev PROGNOZA POTROŠNJE) ---

def make_prediction(input_data: pd.DataFrame):
    """
    Pravi prognozu za N dana koristeći NN i RFR model.
    """
    # 1. Učitavanje resursa (Poziv Data Services)
    model, rfr_model, scaler_X, scaler_y, X_template = get_ml_resources()

    if model is None:
        return "Modeli nisu pronađeni. Prvo pokrenite trening.", pd.DataFrame()

    # 2. FEATURE ENGINEERING (Isto kao u treningu!)
    
    input_data.index = pd.to_datetime(input_data.index)
    
    # Kreiranje vremenskih karakteristika (hour, day_of_week, month, year)
    input_data['hour'] = input_data.index.hour
    input_data['day_of_week'] = input_data.index.dayofweek
    input_data['month'] = input_data.index.month
    input_data['year'] = input_data.index.year
    input_data['day_of_year'] = input_data.index.dayofyear
    
    # Ciklično kodiranje
    input_data = encode_cyclic_feature(input_data, 'hour', 24)
    input_data = encode_cyclic_feature(input_data, 'day_of_week', 7)
    input_data = encode_cyclic_feature(input_data, 'month', 12)
    input_data = encode_cyclic_feature(input_data, 'day_of_year', 366)

    
    # One-Hot Encoding
    input_data = pd.get_dummies(input_data, columns=['conditions'], prefix='cond', drop_first=True)
    
    # 3. Usklađivanje kolona sa trening setom
    # Koristi se lista kolona sačuvana pri treningu
    X_pred = input_data.reindex(columns=X_template.columns.tolist(), fill_value=0)
    # 4. Skaliranje i priprema za NN (ROBUSTNA METODA IZ STAROG KODA)
    numerical_cols_to_scale = COLS_TO_SCALE  # Lista kontinuiranih kolona
    X_pred_all_cols = X_pred.copy() # X_pred je neskaliran i služi za RFR

    # 4.1. Izdvajanje podataka
    # Moramo obezbediti da su kolone u X_pred numeričke pre nego što ih prosledimo skaleru!
    X_pred_to_scale = X_pred_all_cols[numerical_cols_to_scale].astype('float64') 
    X_pred_unscaled = X_pred_all_cols.drop(columns=numerical_cols_to_scale) # Sve ostale kolone

    # 4.2. Skaliranje numeričkih podataka
    # Transformišemo samo numeričke kolone
    X_pred_scaled_data = scaler_X.transform(X_pred_to_scale)

    # 4.3. Stvaranje finalnog, skaliranog DataFrame-a za NN (KORISTIMO CONCAT)
    X_pred_scaled = pd.concat([
        pd.DataFrame(X_pred_scaled_data, columns=numerical_cols_to_scale, index=X_pred_to_scale.index),
        X_pred_unscaled
    ], axis=1)

    # 4.4. Usklađivanje redosleda kolona
    # X_template_cols je lista kolona sačuvana pri treningu
    X_pred_scaled = X_pred_scaled[X_template.columns.tolist()] # X_template se učitava u get_ml_resources()
    X_pred_scaled = X_pred_scaled.astype('float32')
    #print('Dovde je proso nakon skaliranja \n', X_pred_scaled.columns)
    
    # 5. Prognoza
    nn_predictions = model.predict(X_pred_scaled).flatten()
    #nn_predictions = scaler_y.inverse_transform(nn_pred_scaled).flatten()
    rfr_predictions = rfr_model.predict(X_pred) # RFR predviđa na neskaliranim podacima

    #print(f"\n--- DIJAGNOSTIKA PREDIKCIJA ---")
    print(f"NN predikcije: min={nn_predictions.min():.2f}, max={nn_predictions.max():.2f}")
    print(f"RFR predikcije: min={rfr_predictions.min():.2f}, max={rfr_predictions.max():.2f}")
    
    realistic_min = 3000  # Minimalna realna potrošnja
    realistic_max = 11000  # Maksimalna realna potrošnja
    valid_predictions = nn_predictions[(nn_predictions >= realistic_min) & (nn_predictions <= realistic_max)]
    if len(valid_predictions) > 0:
        mean_prediction = np.mean(valid_predictions)
    else:
        mean_prediction = (realistic_min + realistic_max) / 2  # fallback
        print(f'Nema validnih predikcija za izračun prosjeka, koristim fallback: {mean_prediction:.2f}')
    
    # LOGOVANJE KOLIKO PREDIKCIJA JE KORIGOVANO - POBOLJŠANJE
    outlier_count = ((nn_predictions < realistic_min) | (nn_predictions > realistic_max)).sum()
    total_predictions = len(nn_predictions)
    print(f'Korigovano {outlier_count}/{total_predictions} predikcija ({outlier_count/total_predictions*100:.1f}%)')
    
    # KORIGOVANJE PREDIKCIJA
    nn_predictions_corrected = np.where(
        (nn_predictions < realistic_min) | (nn_predictions > realistic_max),
        mean_prediction,
        nn_predictions
    )
    
    print(f'Originalne NN predikcije: min={nn_predictions.min():.2f}, max={nn_predictions.max():.2f}')
    print(f'Korigovane NN predikcije: min={nn_predictions_corrected.min():.2f}, max={nn_predictions_corrected.max():.2f}')
    # Koristite korigovane predikcije
    rfr_predictions = rfr_model.predict(X_pred)
    
    # 6. Formatiranje rezultata
    results = pd.DataFrame({
        'timestamp': input_data.index,
        'NN_Prediction (kWh)': nn_predictions_corrected,  # KORIGOVANE PREDIKCIJE
        'RFR_Prediction (kWh)': rfr_predictions
    }).set_index('timestamp')
    
    # 7. Čuvanje rezultata
    engine = create_db_engine()
    insert_prediction_result(results.reset_index(), engine)
    
    return "Prognoza uspešno izvršena. Rezultati sačuvani u bazi.", results