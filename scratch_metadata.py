import requests
import os
import json
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
from time import sleep

pars = ["air_temperature", "extreme_temperature", "extreme_wind", "precipitation", "solar", "wind"]
folds = ["air_temp", "ex_temp", "ex_wind", "precip", "solar", "wind"]

def get_names(par, fold):
    url = f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{par}/meta_data/"
    page_to_scrape = requests.get(url)
    soup = BeautifulSoup(page_to_scrape.text, "html.parser")

    file_names = [
        link.get("href") 
        for link in soup.find_all("a") 
        if link.get("href") and link.get("href").endswith(".zip")
    ]

    with open(f"data/metadata/names.txt", "w") as f:
        f.write("\n".join(file_names))

    print(f"Got names for {fold}!")
    return file_names

# scrape zip files
def get_zips(par, fold):
    file_names = get_names(par, fold)

    base_url = f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{par}/meta_data/"

    os.makedirs(f"data/metadata/zips", exist_ok=True)

    for name in file_names:
        url = base_url + name
        response = requests.get(url, timeout=30)

        with open(f"data/metadata/zips/{name}", "wb") as f:
            f.write(response.content)

    print(f"Got files for {fold}!")

def extract_zips(fold):
    zip_folder = f"data/metadata/zips"
    unzip_folder = f"data/metadata/unzips"

    os.makedirs(unzip_folder, exist_ok=True)

    for f in os.listdir(zip_folder):
        zip_path = os.path.join(zip_folder, f)
        with ZipFile(zip_path, 'r') as zip_ref:
            file = zip_ref.namelist()[0]
            path = os.path.join(unzip_folder, file)
            if not os.path.exists(path):
                zip_ref.extract(file, unzip_folder)

    print(f"Unzipped metadata for {fold}!")

if __name__ == "__main__":
    print("---------- Starting Metadata Scraping Workflow ----------")
        
    path = Path(f"plots")
    path.mkdir(parents=True, exist_ok=True)
    print("\nAll required directories ensured.")

    for par, fold in zip(pars, folds):
        get_names(par, fold)
        
    print("\nStarting zips download.")
    for par, fold in zip(pars, folds):
        get_zips(par, fold)

    print("\nStarting file extraction.")
    for fold in folds:
        extract_zips(fold)
    
    print("\n---------- Workflow Complete ----------")