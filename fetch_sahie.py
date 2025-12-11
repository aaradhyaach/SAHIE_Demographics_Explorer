# fetch_sahie.py
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from time import sleep

# Load variables from the .env file into the environment
load_dotenv() 

# Access the variable just like a system environment variable
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("API_KEY not found. Please set it in your .env file or environment.")

API_BASE = "https://api.census.gov/data/timeseries/healthins/sahie"

def fetch_year_state_race(year, state_fips):
    params = {
        "get": "YEAR,RACE_DESC,RACECAT,PCTUI_PT,PCTUI_LB90,PCTUI_UB90,NAME,STATE",
        "for": f"state:{state_fips}",
        "time": str(year),
        "key": API_KEY
    }
    r = requests.get(API_BASE, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    cols = data[0]
    rows = data[1:]
    return pd.DataFrame(rows, columns=cols)

def fetch_range_state_race(years, state_fips, pause=0.3):
    frames = []
    for y in years:
        df = fetch_year_state_race(y, state_fips)
        frames.append(df)
        sleep(pause)  
    big = pd.concat(frames, ignore_index=True, sort=False)
    # convert numeric columns
    for c in ["PCTUI_PT","PCTUI_LB90","PCTUI_UB90","YEAR","STATE"]:
        if c in big.columns:
            big[c] = pd.to_numeric(big[c], errors="coerce")
    return big

if __name__ == "__main__":
    years = list(range(2010, 2022))  # adjust as needed
    state_fips = "06"  # California example
    df = fetch_range_state_race(years, state_fips)
    print(df.head())
    df.to_csv("sahie_state06_by_race_2010_2022.csv", index=False)
