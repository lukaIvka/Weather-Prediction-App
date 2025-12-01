import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# --- KORAK 1: UČITAVANJE MODELA I SKALERA ---
# Putanje do sačuvanih fajlova
MODELS_PATH = Path('C:/Users/User/Desktop/ISISproj/isis_projekat/models/')
@st.cache_resource
def get_resources():
    try:
        # Učitavanje skalera. Oni su neophodni da bi novi podaci bili skalirani na isti način kao podaci za trening.
        scaler_X = joblib.load(MODELS_PATH / 'scaler_X.pkl')
        scaler_y = joblib.load(MODELS_PATH / 'scaler_y.pkl')
        # Učitavanje istreniranog modela
        model = load_model(MODELS_PATH / 'model.keras') #ovdje nakon pokretanja train_model.py promijeniti u 'model.keras'
        rfr_model = joblib.load(MODELS_PATH / 'rfr_model.pkl')
        # Učitavanje testnog seta
        X_test_final = joblib.load(MODELS_PATH / 'X_test_final.pkl')
        y_test_scaler = joblib.load(MODELS_PATH / 'y_test_scaler.pkl')
        print("Model i skaleri su uspešno učitani.")
        return scaler_X, scaler_y, model, X_test_final, y_test_scaler, rfr_model
    except FileNotFoundError:
        print(f"Greška: Fajlovi modela ili skalera nisu pronađeni. Proveri da li se nalaze na putanji {MODELS_PATH}")
        # Ako fajlovi ne postoje, program se zaustavlja
        exit()
      
def encode_cyclic_feature_predict(df, col, max_val):
    """Primenjuje Sin/Cos transformaciju na jedan red DataFrame-a."""
    df_copy = df.copy() # <--- KRITIČNA KOPIJA
    df_copy[col + '_sin'] = np.sin(2 * np.pi * df_copy[col] / max_val)
    df_copy[col + '_cos'] = np.cos(2 * np.pi * df_copy[col] / max_val)
    df_copy = df_copy.drop(columns=[col])
    return df_copy

def evaluate_and_visualize(scaler_X, scaler_y, model, X_test_final, y_test_scaler):
    # --- KORAK 2: PRIPREMA NOVOG ULAZNOG PRIMERA ---
    # Ovde bi trebalo da uneseš nove podatke za koje želiš predviđanje.
    # Za demonstraciju, uzećemo jedan red iz testnog seta, ali u praksi bi to bio neki novi podatak.
    # Podatke treba pripremiti na POTPUNO ISTI NAČIN kao i podatke za trening (dodati sve kolone, skalirati...).
    # Tvoj kod za pripremu podataka ovde bi bio isti kao u train_model.py
    # Pretpostavićemo da imaš X_test_final iz tvog prethodnog fajla.

    # Učitavanje testnih podataka za predviđanje
    #X_test_final = pd.read_pickle('C:/Users/User/Desktop/ISISproj/isis_projekat/models/X_test_final.pkl')
    #y_test_scaler = pd.read_pickle('C:/Users/User/Desktop/ISISproj/isis_projekat/models/y_test_scaler.pkl')

    # --- KORAK 3: PRAVLJENJE PREDVIĐANJA ---
    # Model predviđa na skaliranim podacima i vraća skalirani rezultat.
    loss, mae = model.evaluate(X_test_final, y_test_scaler, verbose=0)
    print("\n--- Evaluacija modela na testnom setu ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    # --- KORAK 4: DEKOMPRESIJA PREDVIĐANJA ---
    predictions_scaled = model.predict(X_test_final)
    #Vraćanje skaliranih vrednosti u originalne jedinice (npr. kWh)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_scaler)

    # --- KORAK 5: VIZUALIZACIJA IPRIKAZ REZULTATA ---
    reslut_df = pd.DataFrame({
        'Stvarne Vrijednosti': y_test_original.flatten(),
        'Predvidjanja': predictions.flatten()
    })

    # graficki prikaz
    plt.figure(figsize=(15,8))
    plt.plot(reslut_df.index, reslut_df['Stvarne Vrijednosti'], label = 'Stvarne Vrijednosti')
    plt.plot(reslut_df.index, reslut_df['Predvidjanja'], label = 'Predvidjanja', alpha = 0.7)
    plt.title('Poredjenje Stvarnih i Predvidjenih Vrijednosti Opterecenja')
    plt.xlabel('Vremenski koraci')
    plt.ylabel('Opterecenje (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show() 


    print("\n--- Poređenje stvarnih i predviđenih vrednosti (prvih 5) ---")
    for i in range(5):
        print(f"Predviđeno: {predictions[i][0]:.2f}, Stvarno: {y_test_original[i][0]:.2f}")

    errors = np.abs(y_test_original.flatten() - predictions.flatten())

    plt.bar(range(len(errors)), errors, alpha = 0.7)
    plt.xlabel('Predictions')
    plt.ylabel('Y_test_original - apsolutna greska u kWh')
    plt.title('Prikaz odstupanja - indeksi predvidjanja')
    plt.show()

    final_df = pd.read_pickle('C:/Users/User/Desktop/ISISproj/isis_projekat/data/processed/final_processed_df.pkl')
    correlation_matrix = final_df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelaciona matrica svih karakteristika')
    plt.show()

def make_single_prediction(input_data,scaler_X, scaler_y, model, X_test_final):
    #samo jedan red
    input_df = pd.DataFrame([input_data])
    input_df['date'] = pd.to_datetime(input_data['date'])
    '''
    input_df['hour'] = input_df['date'].dt.hour
    input_df['day_of_week'] = input_df['date'].dt.day_of_week
    input_df['month'] = input_df['date'].dt.month
    input_df['day_of_year'] = input_df['date'].dt.day_of_year
    input_df = input_df.drop(columns=['date'])
    
    input_df = pd.get_dummies(input_df, columns=['conditions'], prefix='conditions')
        
    X_template = pd.DataFrame(0.0, index=[0], columns=X_test_final.columns) 
    for col in input_df.columns:
        if col in X_template.columns:
        # --- DODAJ EKSLICITNU KONVERZIJU ---
        # Konvertuj vrednost u float pre prebacivanja, ako je numerička
            value = input_df.loc[0, col]
            if isinstance(value, (bool, int)):
                value = float(value)
            elif pd.api.types.is_numeric_dtype(value):
                value = float(value)
        
            X_template.loc[0, col] = value
            
    numerical_cols_to_scale = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                               'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                               'solarradiation', 'solarenergy', 'uvindex'] 
    
    X_template[numerical_cols_to_scale] = scaler_X.transform(X_template[numerical_cols_to_scale])
    X_final_single = pd.DataFrame(X_template, columns=X_test_final.columns)
    
    for col in X_final_single.columns:
        if X_final_single[col].dtype != np.float64: # Proveri da li je već float
            X_final_single[col] = X_final_single[col].astype(np.float64) 
    '''
    # Prvo, osigurajmo da je 'is_holiday' DataFrame kolona
  
    input_df['is_holiday'] = input_df['is_holiday'].astype(float)
    
    # 1. Kreiranje vremenskih kolona
    input_df['hour'] = input_df['date'].dt.hour
    input_df['day_of_week'] = input_df['date'].dt.dayofweek
    input_df['month'] = input_df['date'].dt.month
    input_df['day_of_year'] = input_df['date'].dt.dayofyear
    input_df = input_df.drop(columns=['date'])

    # 2. Ciklično kodiranje
    input_df = encode_cyclic_feature_predict(input_df, 'hour', 24)
    input_df = encode_cyclic_feature_predict(input_df, 'day_of_week', 7)
    input_df = encode_cyclic_feature_predict(input_df, 'month', 12)
    input_df = encode_cyclic_feature_predict(input_df, 'day_of_year', 366)

    # 3. One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['conditions'], prefix='conditions')

    # 4. Kreiranje predloška (template)
    X_template = pd.DataFrame(0.0, index=[0], columns=X_test_final.columns) 
    for col in input_df.columns:
        if col in X_template.columns:
            # Pobrini se da se vrednosti prenesu ispravno (kao float)
            X_template.loc[0, col] = input_df.loc[0, col]

    # 5. Određivanje koje kolone skalirati (Isto kao u train_model.py)
    # Ovde koristimo fiksnu listu baziranu na tvojim podacima
    numerical_cols_to_scale = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                               'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                               'solarradiation'] # OVO MORA DA BUDE ISTO KAO U TRAIN_MODEL.PY

    # 6. Skaliranje samo kontinuiranih kolona
    X_to_scale = X_template[numerical_cols_to_scale]
    X_scaled = scaler_X.transform(X_to_scale)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols_to_scale, index=X_template.index)

    # 7. Spajanje nazad (skalirani kontinuirani + neskalirani ostali)
    X_unscaled_cols = [col for col in X_template.columns if col not in numerical_cols_to_scale]
    X_unscaled_df = X_template[X_unscaled_cols]

    # Finalni ulazni red
    X_final_input = pd.concat([X_scaled_df, X_unscaled_df], axis=1)

    # Sortiranje kolona da odgovaraju poretku iz trening seta
    X_final_input = X_final_input[X_test_final.columns]
    
    # Ovo osigurava da Keras dobija ispravan dtype
    for col in X_final_input.columns:
        if X_final_input[col].dtype != np.float64:
            X_final_input[col] = X_final_input[col].astype(np.float64)
    
    print("--------------------------------------------------")
    print(f"Model: {'NN' if 'model' in locals() else 'RFR'}")
    
    # Najvažniji ispis: Vrednost is_holiday tik pre predviđanja
    print(f"VREDNOST is_holiday: {X_final_input['is_holiday'].iloc[0]}")
    
    # Ispis prvih 5 kolona i poslednjih 5 kolona da vidimo skaliranje i ciklično kodiranje
    print("Prvih 5 kolona:")
    print(X_final_input.iloc[0, :5].to_string())
    print("Poslednjih 5 kolona:")
    print(X_final_input.iloc[0, -5:].to_string())
    print("--------------------------------------------------")
    
    predicted_scaled = model.predict(X_final_input.values)
    predicted_load_values = scaler_y.inverse_transform(predicted_scaled)[0][0]
    
    return predicted_load_values

def make_rfr_prediction(input_data, rfr_model, scaler_X, X_test_final):
    input_df = pd.DataFrame([input_data])
    input_df['date'] = pd.to_datetime(input_data['date'])
    '''
    input_df['hour'] = input_df['date'].dt.hour
    input_df['day_of_week'] = input_df['date'].dt.day_of_week
    input_df['month'] = input_df['date'].dt.month
    input_df['day_of_year'] = input_df['date'].dt.day_of_year
    input_df = input_df.drop(columns=['date'])
    
    input_df = pd.get_dummies(input_df, columns=['conditions'], prefix='conditions')
        
    X_template = pd.DataFrame(0.0, index=[0], columns=X_test_final.columns) 
    for col in input_df.columns:
        if col in X_template.columns:
        # --- DODAJ EKSLICITNU KONVERZIJU ---
        # Konvertuj vrednost u float pre prebacivanja, ako je numerička
            value = input_df.loc[0, col]
            if isinstance(value, (bool, int)):
                value = float(value)
            elif pd.api.types.is_numeric_dtype(value):
                value = float(value)
        
            X_template.loc[0, col] = value
            
    numerical_cols_to_scale = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                               'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                               'solarradiation', 'solarenergy', 'uvindex'] 
    
    X_template[numerical_cols_to_scale] = scaler_X.transform(X_template[numerical_cols_to_scale])
    X_final_single = pd.DataFrame(X_template, columns=X_test_final.columns)
    '''
    # Prvo, osigurajmo da je 'is_holiday' DataFrame kolona
  
    input_df['is_holiday'] = input_df['is_holiday'].astype(float)
    
        # 1. Kreiranje vremenskih kolona
    input_df['hour'] = input_df['date'].dt.hour
    input_df['day_of_week'] = input_df['date'].dt.dayofweek
    input_df['month'] = input_df['date'].dt.month
    input_df['day_of_year'] = input_df['date'].dt.dayofyear
    input_df = input_df.drop(columns=['date'])
    
    # 2. Ciklično kodiranje
    input_df = encode_cyclic_feature_predict(input_df, 'hour', 24)
    input_df = encode_cyclic_feature_predict(input_df, 'day_of_week', 7)
    input_df = encode_cyclic_feature_predict(input_df, 'month', 12)
    input_df = encode_cyclic_feature_predict(input_df, 'day_of_year', 366)
    
    # 3. One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['conditions'], prefix='conditions')
    
    # 4. Kreiranje predloška (template)
    X_template = pd.DataFrame(0.0, index=[0], columns=X_test_final.columns) 
    for col in input_df.columns:
        if col in X_template.columns:
            # Pobrini se da se vrednosti prenesu ispravno (kao float)
            X_template.loc[0, col] = input_df.loc[0, col]
            
    # 5. Određivanje koje kolone skalirati (Isto kao u train_model.py)
    # Ovde koristimo fiksnu listu baziranu na tvojim podacima
    numerical_cols_to_scale = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                               'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                               'solarradiation'] # OVO MORA DA BUDE ISTO KAO U TRAIN_MODEL.PY
    
    # 6. Skaliranje samo kontinuiranih kolona
    X_to_scale = X_template[numerical_cols_to_scale]
    X_scaled = scaler_X.transform(X_to_scale)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_cols_to_scale, index=X_template.index)
    
    # 7. Spajanje nazad (skalirani kontinuirani + neskalirani ostali)
    X_unscaled_cols = [col for col in X_template.columns if col not in numerical_cols_to_scale]
    X_unscaled_df = X_template[X_unscaled_cols]
    
    # Finalni ulazni red
    X_final_input = pd.concat([X_scaled_df, X_unscaled_df], axis=1)
    
    # Sortiranje kolona da odgovaraju poretku iz trening seta
    X_final_input = X_final_input[X_test_final.columns]
    
    # Ovo osigurava da Keras dobija ispravan dtype
    for col in X_final_input.columns:
        if X_final_input[col].dtype != np.float64:
            X_final_input[col] = X_final_input[col].astype(np.float64)
    
    print("--------------------------------------------------")
    print(f"Model: {'NN' if 'model' in locals() else 'RFR'}")
    
    # Najvažniji ispis: Vrednost is_holiday tik pre predviđanja
    print(f"VREDNOST is_holiday: {X_final_input['is_holiday'].iloc[0]}")
    
    # Ispis prvih 5 kolona i poslednjih 5 kolona da vidimo skaliranje i ciklično kodiranje
    print("Prvih 5 kolona:")
    print(X_final_input.iloc[0, :5].to_string())
    print("Poslednjih 5 kolona:")
    print(X_final_input.iloc[0, -5:].to_string())
    print("--------------------------------------------------")
    
    rfr_prediction = rfr_model.predict(X_final_input.values)
    return rfr_prediction[0]
    
    