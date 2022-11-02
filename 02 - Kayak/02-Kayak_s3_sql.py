# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:25:10 2022

@author: Yann
"""

import boto3
import pandas as pd

# Saving under S3
print("\n# Saving Kayak csv files under S3")
BUCKET_NAME = 'ylequere-jedha'
s3 = boto3.resource("s3")
bucket = s3.Bucket(BUCKET_NAME)

# URL : https://ylequere-jedha.s3.eu-west-3.amazonaws.com/02-Kayak/locations_weather.csv
print("# URL : https://ylequere-jedha.s3.eu-west-3.amazonaws.com/02-Kayak/locations_weather.csv")
bucket.upload_file(r'.\df_locations_weather.csv', Key='02-Kayak/locations_weather.csv')
# URL : https://ylequere-jedha.s3.eu-west-3.amazonaws.com/02-Kayak/hotels.csv
print("# URL : https://ylequere-jedha.s3.eu-west-3.amazonaws.com/02-Kayak/hotels.csv")
bucket.upload_file(r'.\df_hotels.csv', Key='02-Kayak/hotels.csv')

# Saving data under MySQL DB in AWS RDS
print("\n# Saving data under MySQL DB in AWS RDS")
from sqlalchemy import create_engine
DBUSER = 'admin'
DBPASS = 'swxKGe4t5y2tr53+'
DBHOST = 'kayak.cw9vzlqtne1z.eu-west-3.rds.amazonaws.com'
PORT = 3306
engine = create_engine(f"mysql+pymysql://{DBUSER}:{DBPASS}@{DBHOST}:{PORT}/", echo=False)

print("\n# Creating kayakdb")
engine.execute("CREATE DATABASE IF NOT EXISTS kayakdb")
engine.execute("USE kayakdb")
engine.execute("DROP TABLE IF EXISTS LOCATIONS_WEATHER")
engine.execute("DROP TABLE IF EXISTS HOTELS")

print("\n# Creating LOCATIONS_WEATHER table in kayakdb")
df = pd.read_parquet(r'.\df_locations_weather.parquet')
try:
    df.to_sql('LOCATIONS_WEATHER', engine, index=False)
# Unexpected error which not prevents to save data in DB
except AttributeError:
    pass

print("\n# Creating HOTELS table in kayakdb")
df = pd.read_parquet(r'.\df_hotels.parquet')
try:
    df.to_sql('HOTELS', engine, index=False)
# Unexpected error which not prevents to save data in DB
except AttributeError:
    pass

join_query = """SELECT L.location, L.pop, L.perceived_temperature, L.display_name, L.lat, L.lon
,  H.title, H.score, H.desc, H.url
FROM LOCATIONS_WEATHER L 
INNER JOIN HOTELS H 
ON L.location=H.location
"""
# Get result of join between location_weather and hotels on location into a list
print("\n# Get result of join between location_weather and hotels on location into a list (first row)")
l_w = engine.execute(join_query).fetchall()
print(l_w[0])
print(f"=> Size of query result : {len(l_w)} rows")

# Get result of join between location_weather and hotels on location into a DataFrame
print("\n# Get result of join between location_weather and hotels on location into a DataFrame (5 first rows)")
df_result = pd.read_sql(join_query, engine)
print(df_result.head())    
print(f"=> Size of query result: {len(df_result.index)} rows")
