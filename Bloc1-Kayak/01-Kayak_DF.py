# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:02:29 2022

@author: Yann
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

pd.options.display.max_columns = 10
pd.options.display.width = 200

locations = [
"Mont Saint Michel",
"St Malo",
"Bayeux",
"Le Havre",
"Rouen",
"Paris",
"Amiens",
"Lille",
"Strasbourg",
"Chateau du Haut Koenigsbourg",
"Colmar",
"Eguisheim",
"Besancon",
"Dijon",
"Annecy",
"Grenoble",
"Lyon",
"Gorges du Verdon",
"Bormes les Mimosas",
"Cassis",
"Marseille",
"Aix en Provence",
"Avignon",
"Uzes",
"Nimes",
"Aigues Mortes",
"Saintes Maries de la mer",
"Collioure",
"Carcassonne",
"Ariege",
"Toulouse",
"Montauban",
"Biarritz",
"Bayonne",
"La Rochelle"
]

# Build locations dataframe
print("Building locations dataframe")
df_locations = pd.DataFrame(data=locations, columns=['location'])
print(f"Location DF contains {len(df_locations.index)} items.")

# Buiding GPS coordinates
print("\nBuilding GPS coordinates")
display_names = []
lon = []
lat = []
for row in df_locations.itertuples():
    location = row.location.replace(' ', '+')
    url = f"https://nominatim.openstreetmap.org/?q={location}&country=France&format=json"
    r = requests.get(url)
    if r.reason == 'OK':
        r_json = r.json()
        if r_json:
            display_names.append(r_json[0]['display_name'])
            lat.append(float(r_json[0]['lat']))
            lon.append(float(r_json[0]['lon']))
        else:
            raise Exception(f'No openstreetmap json for {location} location')
    else:
        raise Exception(url, "NOK !")

# Adding GPS and display name to main Dataframe
df_locations['display_name'] = display_names
df_locations['lat'] = lat
df_locations['lon'] = lon

# Display Dataframe result
df_locations.to_parquet(r'.\df_locations.parquet')
# Saving as csv for exercice
df_locations.to_csv(r'.\df_locations.csv', index_label='Id', encoding='utf-8')
print(df_locations.head())

# Building weather dataframe
print("\nBuilding weather dataframe")
df_weather = None
for row in df_locations.itertuples():
    location = row.location
    lon = row.lon
    lat = row.lat
    print(f"Getting weather for {location} location")

    API_KEY = "32c093b568434c56f5ee763fa5341101"
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&lang=fr&appid={API_KEY}"
    r = requests.get(url)
    if r.reason != 'OK':
        raise Exception(url, "NOK !")
        
    r_json = r.json()
    if not r_json:
        raise Exception(f'No openweathermap json for {location} location')

    df_local_weather = pd.json_normalize(r_json['list'])
    # We keep only weather at noon, not every 3h
    df_local_weather = df_local_weather[df_local_weather['dt_txt'].str.contains('12:00:00')]
    # We remove unwanted columns
    df_local_weather.drop(columns=['dt', 'weather', 'dt_txt', 'visibility', 'main.temp', 'main.temp_min', 'main.temp_max', 'main.pressure', 'main.sea_level', 'main.grnd_level', 'main.humidity',
                             'main.temp_kf', 'clouds.all', 'wind.speed', 'wind.deg', 'wind.gust', 'sys.pod', 'description', 'rain.3h'], inplace=True, errors='ignore')
    df_local_weather.rename(columns={'main.feels_like':'perceived_temperature'}, inplace = True)
    # We add the location for local weather
    df_local_weather['location'] = location
    # Concatenation of the weather for all locations
    df_weather = (df_local_weather if df_weather is None else pd.concat([df_weather, df_local_weather], ignore_index=True))

df_weather = df_weather.groupby('location', as_index=False)[['pop', 'perceived_temperature']].mean()
df_weather['pop'] = df_weather['pop'].apply(lambda x : int(round(x, 2)*100))
df_weather['perceived_temperature'] = df_weather['perceived_temperature'].apply(lambda x : int(x))
print(df_weather.head())
print(f"Weathers DF contains {len(df_weather.index)} items, with mean weather at noon of 5 next days.")

# Merging locations and weathers
print("\nMerging locations and weathers")
df_locations_weathers = pd.merge(df_weather, df_locations, on='location')
df_locations_weathers.to_parquet(r'.\df_locations_weather.parquet')
# Saving as csv for exercice
df_locations_weathers.to_csv(r'.\df_locations_weather.csv', index_label='Id', encoding='utf-8')
print(df_locations_weathers.head())
print(f"Merged DF contains {len(df_locations_weathers.index)} items, with mean weather at noon of 5 next days.")

# Building hotels dataframe by scraping booking.com
print("\nBuilding hotels dataframe")
df_hotels = None
session = requests.Session()
for row in df_locations.itertuples():
    location = row.location

    print(f"\nGetting hotels for Location {location} on booking.com")
    lon = row.lon
    lat = row.lat    
    url = f"https://www.booking.com/searchresults.fr.html?place_id_lat={lat}&place_id_lon={lon}&order=score&nflt=distance%3D5000"
    r = session.get(url, headers={"Accept-Language" : "fr,en-US;q=0.7,en;q=0.3", "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0)"})
    if r.reason != 'OK':
        raise Exception(url, "NOK !")
    soup = BeautifulSoup(r.text, features="lxml")
    
    titles = []
    for m in soup.findAll("div", class_="fcab3ed991 a23c043802"):
        titles.append(m.text)
        
    if titles == []:
        raise Exception(f'titles not found for {location} location')
    
    scores = []
    for m in soup.findAll("div", class_="b5cd09854e d10a6220b4"):
        scores.append(float(m.text.replace(',','.')))

    if scores == []:
        raise Exception(f'scores not found for {location} location')
    
    urls_hotels = []
    for m in soup.findAll("a", class_="e13098a59f"):
        urls_hotels.append(m.attrs['href'].split('?')[0])

    if urls_hotels == []:
        raise Exception(f'urls hotels not found for {location} location')
    
    desc = []
    for m in soup.findAll("div", class_="a1b3f50dcd f7c6687c3d ef8295f3e6"):
        n = m.findChild("div", class_="d8eab2cf7f", recursive=False)
        if n:
            desc.append(n.text)

    if desc == []:
        raise Exception(f'desc not found for {location} location')
    
    gps = []
    print(f"{len(urls_hotels)} best hotels found !")
    print(f"Getting hotels data for {location} location")
    for url_hotel in urls_hotels:
        r = session.get(url_hotel, headers={"Accept-Language" : "fr,en-US;q=0.7,en;q=0.3", "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0)"})
        soup = BeautifulSoup(r.text, features="lxml")
        latlng = soup.find("a", id="hotel_address").attrs['data-atlas-latlng']
        gps.append({'lat' : float(latlng.split(',')[0]), 'lon' : float(latlng.split(',')[1])})
        # Sleeping one second in order not to be banned by booking.com
        # time.sleep(1)

    if gps == []:
        raise Exception(f'GPS not found for {location} location')

    df_local_hotel = pd.DataFrame(data={'title':titles, 'score':scores, 'desc':desc, 'url':urls_hotels, 'lat':[x ['lat'] for x in gps], 'lon':[x ['lon'] for x in gps]})
    df_local_hotel['location'] = location

    # Concatenation of the weather for all locations
    df_hotels = (df_local_hotel if df_hotels is None else pd.concat([df_hotels, df_local_hotel], ignore_index=True))
    # Hotels are saved temporarily in case of failure in the middle of process
    df_hotels.to_parquet(r'.\df_hotels.parquet')
df_hotels.to_csv(r'.\df_hotels.csv', index_label='Id',encoding='utf-8')
