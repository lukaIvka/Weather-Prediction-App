import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

def load_and_clean_data():

    #postavljanje boljeg displaya dataframea
    #BOLJE ORGANIZUJ TRY EXCEPT BLOKOVE DA SE PISU GRESKE NORMALNIJE GDJE TREBAJU A NE SAMO FILENOTFOUND

    #kreiranje promjenjljivih za nazive datoteka
    holidays_path = r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\US Holidays 2018-2021.xlsx"
    weather_paths = (r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\NYS Weather Data\New York City, NY\New York City, ... 2018-01-01 to 2018-12-31.csv",
                     r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\NYS Weather Data\New York City, NY\New York City, ... 2019-01-01 to 2019-12-31.csv",
     r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\NYS Weather Data\New York City, NY\New York City, ... 2020-01-01 to 2020-12-31.csv",
     r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\NYS Weather Data\New York City, NY\New York City, ... 2021-01-01 to 2021-12-11.csv")

    load_directory = Path(r"C:\Users\User\Desktop\ISISproj\isis_projekat\data\raw\NYS Load  Data")

    #ovo su folderi sa csv fajlovima
    data_folders = [item for item in load_directory.iterdir() if item.is_dir()]
    data_csv = list( load_directory.rglob('*.csv'))
    # print(len(data_folders))
    #print(len(data_csv))

    #pd.set_option('display.max_rows', None)  # Show all rows
    #pd.set_option('display.max_columns', None)  # Show all columns
    #pd.set_option('display.width', None)  # Adjust width for better readability

    try:
        holiday_df = pd.read_excel(holidays_path)
        drop_index = list(range(17, len(holiday_df), 18))
        holiday_frame = holiday_df.drop(drop_index).drop(holiday_df.columns[0], axis=1).reset_index(drop=True)
        holiday_frame = holiday_frame.rename(columns={"Unnamed: 1":"Day","Unnamed: 2":"Date","Unnamed: 3":"Holiday"})
        #print(holiday_df.head)
        #formating data frame, dropping first column and every 18th, and dropping first row
    # see results 


    # for iterating through rows in dataframes
    #for rows in holiday_frame.itertuples():
    #    print(f'coulums:', {rows.Holiday})

    #checking date format and column type
    #print(holiday_frame['Date'].head())
    #print(holiday_frame['Date'].dtype)

    except FileNotFoundError:
        print(f"Greška: Fajl nije pronađen na putanji: {holidays_path}")
        print("Proverite da li je putanja tačna i da li fajl postoji.")
    
    #header = pd.DataFrame["name","datetime","temp","feelslike","dew","humidity","precip","precipprob","preciptype","snow","snowdepth","windgust","windspeed","winddir","sealevelpressure","cloudcover","visibility","solarradiation","solarenergy","uvindex","severerisk","conditions"]  
    
    weather_dfs = []    
    try:
        for item in weather_paths:
            weather_data = pd.read_csv(item)
            weather_dfs.append(pd.DataFrame(weather_data))
            all_weather_dfs = pd.concat(weather_dfs, ignore_index=True)
        all_weather_dfs['datetime'] = pd.to_datetime(all_weather_dfs['datetime'])
        all_weather_dfs = all_weather_dfs.set_index('datetime').sort_index()

        abs_temp = abs(all_weather_dfs['temp'] - all_weather_dfs['feelslike'])
        #check for abnormalities
        date_error = ((all_weather_dfs.index > pd.to_datetime('2022-1-1')) | (all_weather_dfs.index < pd.to_datetime('2018-1-1')))
        if  (date_error.any() != False):
            print('postoji greska u datumu \n')
        if(all_weather_dfs['temp'].any() >120 or all_weather_dfs['temp'].any()<= -20 ):
            print('postoji greska u temperaturi \n')
        if(all_weather_dfs['humidity'].any() >100 or all_weather_dfs['humidity'].any() <0):
            print('greska u humidity\n')
        if(abs_temp.any()>25):
            print('greska u feelslike\n')
        if(all_weather_dfs['windspeed'].any() < 0 | all_weather_dfs['windspeed'].any()>100):
            print('greska u windspeed \n')
        if(all_weather_dfs['precip']).any() < 0:
            print('greska u precip\n ')
        if(all_weather_dfs['snow']).any() < 0:
            print('greska u snow\n')
        if(all_weather_dfs['snowdepth']).any() < 0:
            print('greska u snowdepth\n')
        if(all_weather_dfs['winddir'].any()< 0 | all_weather_dfs['winddir'].any()>360):
            print('greska u winddir\n') 
        if(all_weather_dfs['sealevelpressure'].any()< 950 | all_weather_dfs['sealevelpressure'].any()>1050):
            print('greska u sealevelpressure')
        if(all_weather_dfs['cloudcover'].any()< 0 | all_weather_dfs['cloudcover'].any()>100):
            print('greska u cloudcover\n')
        if(all_weather_dfs['visibility'].any()< 0):
            print('greska u visibility\n')
        if(all_weather_dfs['uvindex'].any()< 0 | all_weather_dfs['uvindex'].any()>1050):
            print('greska u uvindex\n')



        double = all_weather_dfs.index.duplicated().sum() #check for duplicates
        #drop empty columns and fill the empty cells
        all_weather_dfs.drop(['precipprob','preciptype','severerisk', 'windgust'], axis=1, inplace=True)
        all_weather_dfs['solarradiation']= all_weather_dfs['solarradiation'].fillna(0.0)     
        all_weather_dfs['solarenergy']= all_weather_dfs['solarenergy'].fillna(0.0)   
        all_weather_dfs['feelslike'] = all_weather_dfs['feelslike'].interpolate(limit_direction="both")
        all_weather_dfs['dew'] = all_weather_dfs['dew'].interpolate(limit_direction="both")
        all_weather_dfs['humidity'] = all_weather_dfs['humidity'].interpolate(limit_direction="both")
        all_weather_dfs['precip'] = all_weather_dfs['precip'].interpolate(limit_direction="both")
        all_weather_dfs['windspeed'] = all_weather_dfs['windspeed'].interpolate(limit_direction="both")
        all_weather_dfs['winddir'] = all_weather_dfs['winddir'].interpolate(limit_direction="both")
        all_weather_dfs['sealevelpressure'] = all_weather_dfs['sealevelpressure'].interpolate(limit_direction="both")
        all_weather_dfs['cloudcover'] = all_weather_dfs['cloudcover'].interpolate(limit_direction="both")
        all_weather_dfs['visibility'] = all_weather_dfs['visibility'].interpolate(limit_direction="both")
        all_weather_dfs['conditions'] = all_weather_dfs['conditions'].ffill()
        all_weather_dfs['conditions'] = all_weather_dfs['conditions'].bfill()
        #all fields filtered and 4 rows dropped



        empty_field = all_weather_dfs.isnull().sum() #check empty fields
        #print( empty_field)
    
    except FileNotFoundError:
        print(f"Greška: Fajl nije pronađen na putanji: {weather_data}")
        print("Proverite da li je putanja tačna i da li fajl postoji.")

    data_df = []
    try:
        for folder, subfolder, files in os.walk(load_directory):
            for csv in files:
                file_path = os.path.join(folder, csv)
                df = pd.read_csv(file_path)
                data_df.append(df)

        data_df = pd.concat(data_df, ignore_index=True)
        drop_data = data_df[data_df['Name']!= 'N.Y.C.'].index
        data_df = data_df.drop(drop_data)
        #display data types of dataframe
        data_df['Time Stamp'] = pd.to_datetime(data_df['Time Stamp'])
        data_df['Load'] = data_df['Load'].ffill()
        data_df['Load'] = data_df['Load'].bfill()
        data_df =data_df.set_index('Time Stamp').sort_index()
        data_df['Load'] = data_df['Load'].resample('h').mean()
        data_df = data_df.dropna()
        empty_load = data_df[data_df.isnull().any(axis =1 )]
        #print('\nweather prvi:', len(all_weather_dfs))   
        #print('\nload prvi:', len(data_df))

        load_maxH = data_df.index.max()
        load_minH = data_df.index.min()
        weather_maxH = all_weather_dfs.index.max()
        weather_minH = all_weather_dfs.index.min()
        date_range_load = pd.date_range(load_minH, load_maxH, freq='h')
        date_range_weather = pd.date_range(weather_minH, weather_maxH, freq='h')

        data_df = data_df.groupby(data_df.index).first()
        all_weather_dfs = all_weather_dfs.groupby(all_weather_dfs.index).first()
        #print('\nweather nakon groupby:', len(all_weather_dfs))
        #print('\load nakon groupby:', len(data_df))
        ### OVDJE TREBA DA SE VIDI DA LI SE KREIRAJU OBA ZA OBA DATAFRAMEA, I DA SE DETALJNO OBJASNI UBACIVANJE KAO INDEKSA U DATAFRAME
        data_df = data_df.reindex(date_range_load)    
        all_weather_dfs = all_weather_dfs.reindex(date_range_weather)
        #duplicates = all_weather_dfs.index[all_weather_dfs.index.duplicated()]
        #print("\nNAN vrijednosti weather:",all_weather_dfs.index)
        #print("\nNAN vrijednosti load:",data_df.index)
        ###PROVJERI NAN VRIJEDNOSTI I REDOVE I KOLONE DATAFRAMEA DA LI NEKE IZBACITI SKROZ
        final_df = pd.merge(all_weather_dfs, data_df, left_index=True, right_index=True)
        final_df['name'] = final_df['name'].ffill()
        final_df['name'] = final_df['name'].bfill()
        final_df['conditions'] = final_df['conditions'].ffill()
        final_df['conditions'] = final_df['conditions'].bfill()
        final_df['Time Zone'] = final_df['Time Zone'].ffill()
        final_df['Time Zone'] = final_df['Time Zone'].bfill()
        final_df['temp'] = final_df['temp'].interpolate(method='linear', limit_direction='both')
        final_df['feelslike'] = final_df['feelslike'].interpolate(method='linear', limit_direction='both')
        final_df['dew'] = final_df['dew'].interpolate(method='linear', limit_direction='both')
        final_df['humidity'] = final_df['humidity'].interpolate(method='linear', limit_direction='both') 
        final_df['precip'] = final_df['precip'].interpolate(method='linear', limit_direction='both')
        final_df['snow'] = final_df['snow'].interpolate(method='linear', limit_direction='both')
        final_df['snowdepth'] = final_df['snowdepth'].interpolate(method='linear', limit_direction='both')
        final_df['windspeed'] = final_df['windspeed'].interpolate(method='linear', limit_direction='both')
        final_df['winddir'] = final_df['winddir'].interpolate(method='linear', limit_direction='both')
        final_df['sealevelpressure'] = final_df['sealevelpressure'].interpolate(method='linear', limit_direction='both')
        final_df['cloudcover'] = final_df['cloudcover'].interpolate(method='linear', limit_direction='both')
        final_df['visibility'] = final_df['visibility'].interpolate(method='linear', limit_direction='both')
        final_df['solarradiation'] = final_df['solarradiation'].interpolate(method='linear', limit_direction='both')
        final_df['solarenergy'] = final_df['solarenergy'].interpolate(method='linear', limit_direction='both')
        final_df['uvindex'] = final_df['uvindex'].interpolate(method='linear', limit_direction='both')
        final_df['PTID'] = final_df['PTID'].interpolate(method='linear', limit_direction='both')
        final_df['Load'] = final_df['Load'].interpolate(method='linear', limit_direction='both')
        #dropping unnecessary columns 
        final_df.drop(['name', 'Time Zone', 'Name', 'PTID'], axis=1, inplace=True)

        #print("\n Praznine: ", final_df.isnull().sum())
        #print("\n zaglavlja: ", final_df.dtypes)
        final_df = final_df.applymap(lambda x: f"{x:.1f}" if isinstance(x, float) else x)
        final_df.to_pickle('C:/Users/User/Desktop/ISISproj/isis_projekat/data/processed/final_df.pkl')
        final_df.to_csv('C:/Users/User/Desktop/ISISproj/isis_projekat/data/processed/final_df.csv')
    except FileNotFoundError:
        print('Error while loading the load data file\n', f'Check the file path{load_directory}\n')
    
    return final_df

load_and_clean_data()