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

def create_folds(fold):
    path = Path(f"data/{fold}")
    path.mkdir(parents=True, exist_ok=True)

    path = Path(f"plots/{fold}")
    path.mkdir(parents=True, exist_ok=True)

def get_names(par, fold):
    url = f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{par}/historical/"
    page_to_scrape = requests.get(url)
    soup = BeautifulSoup(page_to_scrape.text, "html.parser")

    file_names = [
        link.get("href") 
        for link in soup.find_all("a") 
        if link.get("href") and link.get("href").endswith(".zip")
    ]

    with open(f"data/{fold}/names.txt", "w") as f:
        f.write("\n".join(file_names))

    print(f"Got names for {fold}!")
    return file_names

# scrape zip files
def get_zips(par, fold):
    file_names = get_names(par, fold)

    base_url = f"https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/{par}/historical/"

    os.makedirs(f"data/{fold}/zips", exist_ok=True)

    for name in file_names:
        url = base_url + name
        response = requests.get(url, timeout=30)

        with open(f"data/{fold}/zips/{name}", "wb") as f:
            f.write(response.content)

    print(f"Got files for {fold}!")

def extract_zips(fold):
    zip_folder = f"data/{fold}/zips"
    unzip_folder = f"data/{fold}/unzips"

    os.makedirs(unzip_folder, exist_ok=True)

    for f in os.listdir(zip_folder):
        zip_path = os.path.join(zip_folder, f)
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_folder)

    print(f"Unzipped files for {fold}!")

def count_files_and_size(fold):
    unzip_folder = f"data/{fold}/unzips"
    total_size_bytes = 0
    file_count = 0

    for root, _, files in os.walk(unzip_folder):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                total_size_bytes += os.path.getsize(file_path)
                file_count += 1

    total_size_gb = total_size_bytes / (1024**3)
    return file_count, total_size_gb


def create_used_storage_plots():
    file_counts = {}
    storage_gbs = {}

    for fold in folds:
        count, size = count_files_and_size(fold)
        file_counts[fold] = count
        storage_gbs[fold] = size

    storage_data = {
        "file_counts": file_counts,
        "storage_gbs": storage_gbs
    }
    with open(f"data/storage_unzips.json", "w") as f:
        json.dump(storage_data, f, indent=4)
    
    folds_list = list(file_counts.keys())
    counts_list = list(file_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(folds_list, counts_list, color='#4CAF50')
    plt.title('Number of Files Extracted per Fold', fontsize=14)
    plt.xlabel('Fold Name', fontsize=12)
    plt.ylabel('File Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/file_counts.png")
    plt.close()
    print("Saved plot 'plots/file_counts.png'")

    sizes_list = list(storage_gbs.values())

    plt.figure(figsize=(10, 6))
    plt.bar(folds_list, sizes_list, color='#FF9800')
    plt.title('Used Storage per Fold', fontsize=14)
    plt.xlabel('Fold Name', fontsize=12)
    plt.ylabel('Used Storage (GB)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/storage_gbs.png")
    plt.close()
    print("Saved plot 'plots/storage_gbs.png'")

if __name__ == "__main__":
    print("---------- Starting Scraping Workflow ----------")

    for fold in folds:
        create_folds(fold)
    print("\nAll required directories ensured.")

    for par, fold in zip(pars, folds):
        get_names(par, fold)
        
    print("\nStarting zips download.")
    for par, fold in zip(pars, folds):
        get_zips(par, fold)

    print("\nStarting file extraction.")
    for fold in folds:
        extract_zips(fold)
        
    print("\nGenerating used storage and number of files plots")
    create_used_storage_plots()
    
    print("\n---------- Workflow Complete ----------")