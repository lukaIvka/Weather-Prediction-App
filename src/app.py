# streamlit_app.py
import streamlit as st
import pandas as pd
from datetime import date
from io import StringIO
import matplotlib.pyplot as plt
import plotly.express as px

# Uvoz funkcija iz Servisnog Sloja (src/services)
from services.data_services import get_training_data, update_actual_results_in_db
from services.model_services import train_and_save_models, make_prediction
# Uvoz iz DAL-a
from database.db_connector import create_db_engine, fetch_date_range_from_db

# --- POMOĆNA FUNKCIJA ZA ČITANJE IZ BAZE (ZA UI PREGLED) ---
def fetch_all_results():
    """Čita sve popunjene rezultate (sa ostvarenjima) za prikaz grešaka."""
    try:
        engine = create_db_engine()
        # Selektujemo samo redove gde je actual_load popunjen
        query = "SELECT timestamp, model_name, predicted_load, actual_load, mape, mae FROM prediction_results WHERE actual_load IS NOT NULL ORDER BY timestamp DESC;"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Greška prilikom čitanja rezultata iz baze: {e}")
        return pd.DataFrame()

# 1. Čitanje opsega
engine = create_db_engine()
min_db_date, max_db_date = fetch_date_range_from_db(engine)

# --- STREAMLIT KONFIGURACIJA ---
st.set_page_config(
    page_title="Prognoza Potrošnje Energije (NN/RFR)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Naslov aplikacije
st.title("⚡ Aplikacija za kratkoročnu prognozu potrošnje EE")
#st.markdown("Višeslojna arhitektura: UI (Streamlit) -> Servisni sloj (Python) -> DAL -> PostgreSQL")


# --- NAVIGACIJA (SIDEBAR) ---
menu = ["Uvoz/Trening Modela", "Prognoza Potrošnje", "Pregled Evaluacije"]
choice = st.sidebar.selectbox("Izaberite funkcionalnost", menu)

# ==============================================================================
# 1. STRANICA: UVOZ I TRENING PODATAKA
# ==============================================================================

if choice == "Uvoz/Trening Modela":
    st.header("1. UVOZ PODATAKA I TRENING MODELA")
    st.subheader("Uvoz podataka u bazu")
    
    # 1.1 UVOZ CSV FUNKCIONALNOST (Zahtev: UVOZ PODATAKA)
    st.info("Da bi omogućili funkciju UVOZ PODATAKA, morate uneti kod za direktan INSERT iz CSV-a u 'training_data' u vaš DAL.")
    
    # Trenutno simlujemo da je CSV konvertovan i prosleđen DAL-u
    
    st.subheader("Trening modela")
    st.write("Odaberite datumski opseg za trening i pokrenite treniranje modela.")
    
    col1, col2 = st.columns(2)
    
    # 2. Primena opsega na widgete
    date_start = col1.date_input("Datum OD (početak treninga)", 
                                 value=min_db_date, # Postavi početnu vrednost na MIN
                                 min_value=min_db_date, # Ograniči na MIN
                                 max_value=max_db_date # Ograniči na MAX
                                 )
    date_end = col2.date_input("Datum DO (kraj treninga)", 
                               value=max_db_date, # Postavi početnu vrednost na MAX
                               min_value=min_db_date, # Ograniči na MIN
                               max_value=max_db_date # Ograniči na MAX
                               )
    if st.button("🚀 POKRENI TRENING"):
        try:
            # Prikaz statusa dok se trening izvršava
            with st.spinner('Trening modela u toku...'):
                # Poziv servisnom sloju
                results = train_and_save_models(date_start, date_end)
            
            st.success(f"✅ Trening uspešno izvršen! {results['status']}")

            # Prikaz performansi na test setu
            st.subheader("Performanse na Test Setu")
            
            col_nn, col_rfr = st.columns(2)
            col_nn.metric("NN MAPE", f"{results['nn_mape']:.2f}%")
            col_rfr.metric("RFR MAPE", f"{results['rfr_mape']:.2f}%")
            
            # Grafički prikaz poređenja
            df_plot = pd.DataFrame({
                'Stvarno Ostvarenje': results['y_test'],
                'NN Predikcija': results['nn_predictions'],
                'RFR Predikcija': results['rfr_predictions']
            })
            df_plot.index = get_training_data(date_start, date_end).index[-len(results['y_test']):]
            start_test = df_plot.index[0].strftime('%Y-%m-%d %H:%M')
            end_test = df_plot.index[-1].strftime('%Y-%m-%d %H:%M')
            fig = px.line(
                df_plot,
                title=f"Poređenje modela na Test Setu ({start_test} do {end_test})",
                labels={'value': 'Potrošnja (kWh)', 'timestamp': 'Vreme'}
            )
            fig.update_xaxes(title_text='Vreme')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Greška prilikom treninga: {e}")

# ==============================================================================
# 2. STRANICA: PROGNOZA POTROŠNJE I EVALUACIJA
# ==============================================================================

elif choice == "Prognoza Potrošnje":
    st.header("2. PROGNOZA POTROŠNJE ENERGIJE")
    
    st.subheader("2.1. Unos Ulaznih Podataka za Prognozu (Zahtev: PROGNOZA POTROŠNJE)")
    
    uploaded_file = st.file_uploader(
        "Učitajte CSV sa nezavisnim podacima ($X_{new}$) za period prognoze (7 dana)",
        type="csv"
    )

    if uploaded_file is not None:
        # Čitanje ulaznih podataka
        uploaded_data = uploaded_file.getvalue().decode("utf-8")
        input_df = pd.read_csv(StringIO(uploaded_data), parse_dates=['datetime'], index_col='datetime')
        st.write("Učitani podaci za prognozu (prvih 5 redova):")
        st.dataframe(input_df.head())
        
        st.markdown("---")
        
        # 2.2. POKRETANJE PROGNOZE
        if st.button("🔮 POKRENI PROGNOZU"):
            with st.spinner('Izvršavanje prognoze i upis u bazu...'):
                # Poziv servisnom sloju
                message, results_df = make_prediction(input_df.copy()) 
            
            if not results_df.empty:
                st.success(f"✅ {message}")
                st.subheader(f"Rezultati Prognoze ({len(results_df)} sati)")
                
                # Prikaz predikcija
                st.dataframe(results_df.head(10)) 
                
                # Grafički prikaz
                results_df.columns = ['NN Predikcija', 'RFR Predikcija']
                fig_pred = px.line(
                    results_df,
                    title="Prognoza Potrošnje (NN vs RFR)",
                    labels={'value': 'Potrošnja (kWh)', 'timestamp': 'Vreme'}
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            else:
                st.error(message)

        # 2.3. UNOS STVARNIH OSTVARENJA I EVALUACIJA (Zahtev: POREĐENJE I GREŠKA)
        st.subheader("2.3. Unos stvarnih ostvarenja ($Y_{actual}$) i Ažuriranje Baze")
        st.warning("Ova funkcija je namenjena za unos ostvarenja nakon isteka perioda prognoze.")

        actual_file = st.file_uploader(
            "Učitajte CSV sa STVARNIM OSTVARENJIMA (kolone: datetime, load_kwh)",
            type="csv"
        )
        
        if actual_file is not None and st.button("📊 IZVRŠI EVALUACIJU (UPDATE BAZE)"):
            with st.spinner('Ažuriranje baze stvarnim ostvarenjima i računanje grešaka...'):
                actual_data = pd.read_csv(StringIO(actual_file.getvalue().decode("utf-8")), parse_dates=['datetime'])
                
                # Poziv servisnom sloju
                update_message = update_actual_results_in_db(actual_data)
                
            if "uspešno ažurirana" in update_message:
                st.success(f"✅ {update_message}")
            else:
                st.warning(f"⚠️ Evaluacija nije izvršena: {update_message}")
                
# ==============================================================================
# 3. STRANICA: PREGLED EVALUACIJE
# ==============================================================================

elif choice == "Pregled Evaluacije":
    st.header("3. PREGLED EVALUACIJE I POREĐENJE GREŠAKA")
    st.write("Prikazuje rezultate poređenja prognoze i stvarnih ostvarenja sa izračunatim greškama (MAPE, MAE).")

    results_df = fetch_all_results()

    if results_df.empty:
        st.info("Nema kompletnih rezultata (sa stvarnim ostvarenjima) za prikaz.")
    else:
        st.subheader("Prikaz Popunjenih Rezultata iz Baze")
        st.dataframe(results_df)

        # Agregacija grešaka po modelu
        error_summary = results_df.groupby('model_name')[['mape', 'mae']].mean().reset_index()
        
        st.subheader("Prosečna Greška (MAPE/MAE)")
        
        col_nn_mape, col_rfr_mape = st.columns(2)
        nn_mape = error_summary[error_summary['model_name'] == 'NN']['mape'].values[0] if 'NN' in error_summary['model_name'].values else 0
        rfr_mape = error_summary[error_summary['model_name'] == 'RFR']['mape'].values[0] if 'RFR' in error_summary['model_name'].values else 0

        col_nn_mape.metric("Prosečni NN MAPE", f"{nn_mape:.2f}%")
        col_rfr_mape.metric("Prosečni RFR MAPE", f"{rfr_mape:.2f}%")

        # Grafički prikaz poređenja (Stvarno vs Predviđeno)
        df_chart = results_df.pivot_table(
            index='timestamp',
            columns='model_name',
            values='predicted_load'
        ).reset_index()
        df_chart['Stvarno Ostvarenje'] = results_df.drop_duplicates(subset=['timestamp'])['actual_load'].values

        fig_final = px.line(
            df_chart,
            x='timestamp',
            y=['Stvarno Ostvarenje', 'NN', 'RFR'],
            title='Poređenje: Prognoza vs Stvarno Ostvarenje',
            labels={'value': 'Potrošnja (kWh)', 'timestamp': 'Vreme'}
        )
        st.plotly_chart(fig_final, use_container_width=True)