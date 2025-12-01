import pandas as pd
import numpy as np
import time
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

MODELS_PATH = Path('C:/Users/User/Desktop/ISISproj/isis_projekat/models/')
# Ručno definisana lista kontinuiranih, numeričkih kolona koje zahtevaju StandardScaler
CONTINUOUS_COLS_TO_SCALE = ['temp', 'feelslike', 'dew', 'humidity', 'precip', 'snow', 'snowdepth',
                           'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
                           'solarradiation']

# load holidays csv 
def load_and_clean_holidays(path):
    """Učitava i čisti podatke o praznicima."""
    holiday_df = pd.read_excel(path, skiprows=1) # Preskačemo prvi red sa praznim zaglavljima
    holiday_df.columns = ['Year', 'Day', 'Date', 'Holiday'] # Ručno dodajemo zaglavlja
    holiday_df = holiday_df.drop('Year', axis=1).dropna() # Brišemo nebitnu kolonu i prazne redove
    holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], errors='coerce') # Konvertujemo u datetime
    holiday_df = holiday_df.dropna(subset=['Date']) # Brišemo nevažeće datume (NaT)
    return holiday_df

final_data = pd.read_pickle('C:/Users/User/Desktop/ISISproj/isis_projekat/data/processed/final_df.pkl')
final_df = pd.DataFrame(final_data)
final_df = final_df.apply(lambda col: pd.to_numeric(col, errors='coerce') if col.name != 'conditions' else col)

holidays_path = Path('C:/Users/User/Desktop/ISISproj/isis_projekat/data/raw/US Holidays 2018-2021.xlsx')
holiday_df = load_and_clean_holidays(holidays_path)
#ovdje se moze ubaciti rad sa sin/cos za ciklicni pristup u zavisnosti od rezultata
#final_df['hours'] = final_df.index.hour
#final_df['day_of_week'] = final_df.index.dayofweek
#final_df['month'] = final_df.index.month
#final_df['day_of_year'] = final_df.index.dayofyear
#dates_np = np.array(holiday_df, dtype='datetime64[D]')  # [D] za dan bez vremena
# --- CIKLIČNO KODIRANJE (SIN/COS) ---

# Funkcija za kreiranje sin i cos komponente
def encode_cyclic_feature(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
    df = df.drop(columns=[col]) # Brišemo originalnu kolonu
    return df

# Kreiramo originalne numeričke kolone iz indeksa
final_df['hour'] = final_df.index.hour
final_df['day_of_week'] = final_df.index.dayofweek
final_df['month'] = final_df.index.month
final_df['day_of_year'] = final_df.index.dayofyear

# Primena cikličnog kodiranja
final_df = encode_cyclic_feature(final_df, 'hour', 24)
final_df = encode_cyclic_feature(final_df, 'day_of_week', 7)
final_df = encode_cyclic_feature(final_df, 'month', 12)
final_df = encode_cyclic_feature(final_df, 'day_of_year', 366) # Koristimo 366 zbog prestupnih godina


# Konvertujte index u isti format
final_df['is_holiday'] = final_df.index.normalize().isin(holiday_df['Date']).astype(int)
#OVDO JE ONE-HOT ENCODING A MOZE I CIKLICNO DA SE ODRADI, PREKO SIN/COS PA VIDJETI STA JE BOLJE
final_df = pd.get_dummies(final_df, columns=['conditions'], prefix='conditions')
final_df = final_df.map(lambda x: int(x) if isinstance(x,bool) else x)
# Ovo rešava grešku "KeyError: ['solarenergy', 'uvindex'] not in index"
# Koristimo errors='ignore' za slučaj da su kolone već uklonjene
final_df = final_df.drop(columns=['solarenergy', 'uvindex'], errors='ignore')
#sprint(final_df.columns)
final_df.to_pickle(r'C:\Users\User\Desktop\ISISproj\isis_projekat\data\processed\final_processed_df.pkl')
#print("nakon dodavanja novih kolona: ", final_df['conditions_Clear'])    
#####PROVJERI DA LI JE TREBALO NA DRUGI NACIN RADITI DUMMY CODE I SKONTAJ VEZANO ZA SIN I COS
#final_df[final_df.select_dtypes(include=['float64']).columns]=final_df.select_dtypes(include=['float64'].astype(np.float32))
final_df = final_df.astype({col: np.float32 for col in final_df.select_dtypes(include='float64').columns})

# Koristite numpy isin za bržu provjeru
#final_df['is_holiday'] = np.isin(index_dates, dates_np)
#print('dummies:', final_df.isnull().sum())
#print("final df:\n", (final_df['is_holiday']==True).sum())
#print("days of the week:\n", day_of_week)
#print("months:\n", month)
#print("days of the year:\n", day_of_year)


X = final_df.drop(columns = 'Load')
y = final_df['Load']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=24)

#print("X_train:\n", type(X_train))
#print("X_test:\n", X_test)
#print("y_train:\n", y_train)
#print("y_test:\n", y_test)

'''
int_columns = X_train.select_dtypes(include = ['int']).columns
#print(int_columns[1])
float_columns = X_train.select_dtypes(include = ['float']).columns
#print("ovo su skalarne kolone: ",float_columns)
# decimalni podaci za analizu
X_train_float = X_train[float_columns]
X_test_float = X_test[float_columns]

# cijelobrojni podaci koji nisu za analizu nego spajanje
X_train_int = X_train[int_columns]
X_test_int = X_test[int_columns]

#skalirani numericki podaci
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_float)
X_test_scaled = scaler_X.transform(X_test_float)

X_train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=float_columns, index = X_train_float.index), X_train_int], axis = 1)
X_test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=float_columns, index = X_test_float.index), X_test_int], axis = 1)
'''


# Kolone koje NE TREBA skalirati (binarne i dummy)
# Binarne kolone su sve one koje si kreirao pomocu get_dummies i is_holiday
binary_dummy_cols_prefixes = ['is_holiday', 'conditions_']
binary_dummy_cols = [col for col in X_train.columns if any(col.startswith(p) for p in binary_dummy_cols_prefixes)]

# Sve ciklične kolone su sada Sin/Cos i NE TREBA ih skalirati!
cyclic_cols = [col for col in X_train.columns if col.endswith('_sin') or col.endswith('_cos')]

# Lista kolona koje NE skaliramo (binarne, dummy i ciklične)
unscaled_cols = binary_dummy_cols + cyclic_cols

# Kolone koje TREBA skalirati (kontinuirani podaci, npr. temperatura, vlaga, pritisak, itd.)
# Izuzimamo kolone koje ne skaliramo
# Kolone koje NE skaliramo (binarne i ciklične)
unscaled_cols = [col for col in X_train.columns if col not in CONTINUOUS_COLS_TO_SCALE]
numerical_cols_to_scale = CONTINUOUS_COLS_TO_SCALE

# Izdvajanje podataka za skaliranje
X_train_to_scale = X_train[numerical_cols_to_scale]
X_test_to_scale = X_test[numerical_cols_to_scale]

# Izdvajanje podataka koji se ne skaliraju
X_train_unscaled = X_train[unscaled_cols]
X_test_unscaled = X_test[unscaled_cols]

# Skaliranje isključivo kontinuiranih numeričkih podataka
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_to_scale)
X_test_scaled = scaler_X.transform(X_test_to_scale)

# Kreiranje finalnog DataFrame-a: (skalirani kontinuirani + neskalirani binarni/ciklični)
X_train_final = pd.concat([
    pd.DataFrame(X_train_scaled, columns=numerical_cols_to_scale, index=X_train_to_scale.index),
    X_train_unscaled
], axis=1)

X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=numerical_cols_to_scale, index=X_test_to_scale.index),
    X_test_unscaled
], axis=1)

# Sortiranje kolona da se poklapaju (neophodno za NN i RFR)
X_train_final = X_train_final[X.columns]
X_test_final = X_test_final[X.columns]



#print("Finalni X_train_final nakon skaliranja :\n", X_train_final.head())
#print("\nFinalni X_test_final nakon skaliranja :\n", X_test_final.head())

#skaliranje za y
'''
scaler_y = StandardScaler()
# pravimo da budu dvodimenzionalni nizovi
y_train_scaler = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test_scaler = scaler_y.transform(y_test.values.reshape(-1,1))
'''
#print("Finalni y_train_scaler nakon skaliranja :\n", y_train_scaler[:5])
#print("\nFinalni y_test_scaler nakon skaliranja :\n", y_test_scaler[:5])
#NOVO
scaler_y = MinMaxScaler() # <-- KRITIČNA PROMENA

y_train_scaler = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test_scaler = scaler_y.transform(y_test.values.reshape(-1,1))

#kreiranje modela
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_final.shape[1],)),  
    Dropout(0.2), # Dodajte Dropout (20% neurona se gasi tokom treninga)
    Dense(64, activation='relu'), # 64 neuranona u skrivenom sloju
    Dropout(0.2), # Dodajte Dropout za regularizaciju
    Dense(1, activation='relu')                    # Output layer 1 neuron
])

    

#Kompajliranje modela
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
start_time = time.time()
# Treniranje modela
#  - za prekid overfitting-a
# Trening modela će sada raditi sa pripremljenim podacima
history= model.fit(X_train_final, y_train_scaler, epochs=1000, batch_size=16, verbose=1, validation_split= 0.2,callbacks=[EarlyStopping(monitor='val_loss', patience=50)])
end_time = time.time()

print(f'\nPotrebno vrijeme za treinranje modela: {end_time - start_time: 2f} sekundi \n')
model.save('C:/Users/User/Desktop/ISISproj/isis_projekat/models/model.keras')

import joblib
joblib.dump(scaler_X, 'C:/Users/User/Desktop/ISISproj/isis_projekat/models/scaler_X.pkl')
joblib.dump(scaler_y, 'C:/Users/User/Desktop/ISISproj/isis_projekat/models/scaler_y.pkl')
joblib.dump(X_test_final.columns.tolist(), 'C:/Users/User/Desktop/ISISproj/isis_projekat/models/X_test_final_cols.pkl')
joblib.dump(y_test_scaler, 'C:/Users/User/Desktop/ISISproj/isis_projekat/models/y_test_scaler.pkl')


# Kreiraj grafikon
plt.figure(figsize=(10, 6))

# Crtaj liniju za training loss
plt.plot(history.history['loss'], label='Training Loss')

# Crtaj liniju za validation loss
plt.plot(history.history['val_loss'], label='Validation Loss')

# Dodaj naslov i labele
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#loss, mae = model.evaluate(X_test_final, y_test_scaler, verbose=1)
##print(f"Test Loss: {loss}, Test MAE: {mae}")
#redictions = model.predict(X_test_final)
#print("Predictions:", predictions)
#org_value = scaler_y.inverse_transform(predictions)
##real_value = scaler_y.inverse_transform(y_test_scaler)
#print("\nPrimjeri predviđanja (prvih 5):")
#for i in range(5):
#    print(f"Predviđanje: {org_value[i][0]:.2f}, Stvarna vrijednost: {real_value[i][0]:.2f}")
print("\n--- Trening Random Forest Regressora ---")
rfr_model = RandomForestRegressor(n_estimators=100, random_state=24, n_jobs=-1)
rfr_model.fit(X_train_final, y_train.values)

#evaluacija
rfr_predictions = rfr_model.predict(X_test_final)
rfr_mae = mean_absolute_error(y_test.values, rfr_predictions)
print(f"RFR Test MAE (neskalirano): {rfr_mae:.4f}")

joblib.dump(rfr_model, 'C:/Users/User/Desktop/ISISproj/isis_projekat/models/rfr_model.pkl')
print("Random Forest model sačuvan.")

