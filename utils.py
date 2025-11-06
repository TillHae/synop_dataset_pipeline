import re
import os

import pandas as pd


def create_df(fold):
    rel_path = f"data/{fold}/unzips"
    ret_val = []
    pattern = r".*_(\d{8})_(\d{8})_(\d+)\.txt$"
    for f in os.listdir(rel_path):
        match = re.match(pattern, f)
        if match:
            start_date, end_date, station_id = match.groups()
            ret_val.append({
                "station_id": station_id,
                "start_date": start_date,
                "end_date": end_date
            })
        else:
            print(f"{fold}: Failed for {f}")

    df = pd.DataFrame(ret_val)

    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y%m%d")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d")

    return df


def create_air_temp(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_tu_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;  QN;FF_10;DD_10;eor\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;  -999;-999;eor\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)
    
def create_ex_temp(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_tx_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;  QN;TX_10;TX5_10;TN_10;TN5_10\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;  -999;  -999;  -999;  -999\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)

def create_ex_wind(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_fx_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;  QN;FX_10;FNX_10;FMX_10;DX_10\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;  -999;  -999;  -999;-999\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)

def create_precip(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_rr_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;QN;RWS_DAU_10;RWS_10;RWS_IND_10\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;-999;  -999;-999\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)

def create_solar(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_sd_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;QN;DS_10;GS_10;SD_10;LS_10\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;  -999;  -999;    -999;-999\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)

def create_wind(fold, start_date, end_date, station_id):
    name = f"data/{fold}/unzips/produkt_zehn_min_ff_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{station_id}.txt"
    
    content = "STATIONS_ID;MESS_DATUM;  QN;FF_10;DD_10;eor\n"
    date = start_date
    delta = pd.Timedelta(minutes=10)
    while date < end_date:
        content += f"{station_id.rjust(11)};{date.strftime('%Y%m%d%H%M')}; -999;  -999;-999;eor\n"
        date += delta

    with open(name, "a") as f:
        f.write(content)

def create_file(fold, start_date, end_date, station_id):
    match fold:
        case "air_temp": create_air_temp(fold, start_date, end_date, station_id)
        case "ex_temp": create_ex_temp(fold, start_date, end_date, station_id)
        case "ex_wind": create_ex_wind(fold, start_date, end_date, station_id)
        case "precip": create_precip(fold, start_date, end_date, station_id)
        case "solar": create_solar(fold, start_date, end_date, station_id)
        case "wind": create_wind(fold, start_date, end_date, station_id)
        case _: print(f"{fold} is unknown!")