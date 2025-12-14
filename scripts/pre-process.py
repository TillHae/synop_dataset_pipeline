import os
from multiprocessing import Pool, cpu_count
import re
import pickle
from datetime import datetime as dt, timedelta, time

import pandas as pd
import matplotlib.pyplot as plt

from utils import *

pars = ["air_temperature", "extreme_temperature", "extreme_wind", "precipitation", "solar", "wind"]
folds = ["air_temp", "ex_temp", "ex_wind", "precip", "solar", "wind"]


# check duplicates

def check_duplicates(fold):
    df = create_df(fold)
    dups = []
    for station_id, group in df.groupby("station_id"):
        group = group.sort_values("start_date").reset_index(drop=True)

        start_dates = group["start_date"]
        end_dates = group["end_date"]
        
        diffs = start_dates.values[1:] - end_dates.values[:-1]
        diffs_seconds = diffs / pd.Timedelta(seconds=1)

        if any(diffs_seconds < 0):
            dups.append({
                "station_id": station_id,
                "start_dates": start_dates,
                "end_dates": end_dates,
                "diffs_seconds": diffs_seconds
            })

    dups = pd.DataFrame(dups)
    
    return dups

def plot_duplicates(all_dups):
    fold_names = list(all_dups.keys())
    dup_file_pct = []
    dup_time_pct = []

    for fold in fold_names:
        df = create_df(fold)
        total_files = len(df)
        dup_file_pct.append(len(all_dups[fold]) / total_files * 100 if total_files > 0 else 0)
        total_seconds_all = sum((group["end_date"].max() - group["start_date"].min()).total_seconds()
                                for _, group in df.groupby("station_id"))

        overlap_seconds = 0
        for _, row in all_dups[fold].iterrows():
            overlap_seconds += -row["diffs_seconds"][row["diffs_seconds"] < 0].sum()

        dup_time_pct.append(overlap_seconds / total_seconds_all * 100 if total_seconds_all > 0 else 0)

    os.makedirs("plots/v1", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(fold_names, dup_file_pct, color='steelblue')
    # plt.title('Proportion of Station Files with Overlapping Time Series', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Files with Duplicates (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/duplicate_files_pct.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(fold_names, dup_time_pct, color='darkorange')
    # plt.title('Temporal Overlap in Station Time Series Data', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Overlapping Time (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/duplicate_time_pct.png", dpi=300)
    plt.close()

def delete_duplicates(fold, dups):
    to_delete = []

    for i, station in dups.iterrows():
        for i in range(1, len(station["start_dates"])):
            if (station["start_dates"][i] - station["end_dates"][i - 1]).total_seconds() < 0:
                if station["end_dates"][i] - station["start_dates"][i] < station["end_dates"][i - 1] - station["start_dates"][i - 1]:
                    to_delete.append(f"data/{fold}/unzips/*_{station['start_dates'][i].strftime('%Y%m%d')}_{station['end_dates'][i].strftime('%Y%m%d')}_{station['station_id']}.txt")
                else:
                    to_delete.append(f"data/{fold}/unzips/*_{station['start_dates'][i - 1].strftime('%Y%m%d')}_{station['end_dates'][i - 1].strftime('%Y%m%d')}_{station['station_id']}.txt")

    for f in to_delete:
        os.system(f"rm {f}")

    print(f"{fold} done!")

print("starting duplicate check")
all_dups = {}
for fold in folds:
    dups = check_duplicates(fold)
    all_dups[fold] = dups
    delete_duplicates(fold, dups)

plot_duplicates(all_dups)


# check missing

def check_missing(fold):
    df = create_df(fold)
    missing = []
    for station_id, group in df.groupby("station_id"):
        group = group.sort_values("start_date").reset_index(drop=True)

        start_dates = group["start_date"]
        end_dates = group["end_date"]
        
        diffs = start_dates.values[1:] - end_dates.values[:-1]
        diffs_seconds = diffs / pd.Timedelta(seconds=1)

        if any(diffs_seconds > 86400):
            missing.append({
                "station_id": station_id,
                "start_dates": start_dates,
                "end_dates": end_dates,
                "diffs_seconds": diffs_seconds
            })

    missing = pd.DataFrame(missing)
    return missing

def plot_missing(total_missing):
    fold_names = list(total_missing.keys())
    miss_file_pct = []
    miss_time_pct = []

    for fold in fold_names:
        df = create_df(fold)
        total_files = len(df)
        miss_file_pct.append(len(total_missing[fold])/total_files*100 if total_files>0 else 0)
        total_seconds_all = sum((group["end_date"].max() - group["start_date"].min()).total_seconds()
                                for _, group in df.groupby("station_id"))

        missing_seconds = 0
        for _, row in total_missing[fold].iterrows():
            missing_seconds += row["diffs_seconds"][row["diffs_seconds"] > 86400].sum()

        miss_time_pct.append(missing_seconds / total_seconds_all * 100 if total_seconds_all>0 else 0)

    plt.figure(figsize=(10, 5))
    plt.bar(fold_names, miss_file_pct, color='steelblue')
    # plt.title('Proportion of Station Files with Data Gaps', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Files with Missing Data (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/missing_files_pct.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(fold_names, miss_time_pct, color='darkorange')
    # plt.title('Temporal Coverage: Missing Data Relative to Total Station Time', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Missing Time (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/missing_time_pct.png", dpi=300)
    plt.close()

def create_missing(fold, missing):
    print(f"creating {len(missing)} file(s) for {fold}!")

    for _, station in missing.iterrows():
        for i in range(1, len(station["start_dates"])):
            if (station["start_dates"][i] - station["end_dates"][i - 1]).total_seconds() > 86400:
                create_file(
                    fold,
                    station["end_dates"][i - 1] + pd.Timedelta(days=1),
                    station["start_dates"][i] - pd.Timedelta(days=1),
                    station["station_id"]
                )

    print(f"{fold} done!")

total_missing = {}
for fold in folds:
    missing = check_missing(fold)
    total_missing[fold] = missing
    create_missing(fold, missing)

plot_missing(total_missing)


# fix incorrect file content

def find_incorrect_data(file):
    df = pd.read_csv(
        file,
        sep=';',
        header=0,
        usecols=["MESS_DATUM"],
    )

    expected_delta = pd.Timedelta(minutes=10)
    
    df['time'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H%M')
    df.drop(columns=['MESS_DATUM'], inplace=True) 
    df['delta'] = df['time'].diff()

    df_to_check = df.iloc[1:]
    missing_intervals = df_to_check[df_to_check['delta'] != expected_delta].copy()
    if missing_intervals.empty:
        results_df = pd.DataFrame()
        total_deltas = pd.Timedelta(seconds=0)
        return pd.DataFrame(), total_deltas
    
    print(f"Found {len(missing_intervals)} missing or incorrect intervals for {file}.")

    results_df = missing_intervals.rename(columns={'time': 'end_time'})
    total_deltas = results_df['delta'].sum()
    
    return results_df, total_deltas

def process_file_check(f, unzip_folder):
    if f.startswith(".") or not os.path.isfile(f"{unzip_folder}/{f}"):
        return {}

    file_path = f"{unzip_folder}/{f}"
    with open(file_path, "rb") as station:
        first_line = station.readline()
        second_line = station.readline().rstrip(b"\n").decode()

        station.seek(-1024, 2)
        last_line = station.read().splitlines()[-1].decode()

    station_id = second_line[:11].strip()
    start_str = second_line[12:24]
    end_str = last_line[12:24]

    start_dt = dt.strptime(start_str, "%Y%m%d%H%M")
    end_dt = dt.strptime(end_str, "%Y%m%d%H%M")

    start_ok = start_dt.hour == 0 and start_dt.minute == 0
    end_ok = end_dt.hour == 23 and end_dt.minute == 50

    parts = f.split("_")
    file_start = dt.strptime(parts[-3], "%Y%m%d")
    file_end = dt.strptime(parts[-2], "%Y%m%d")

    expected_start_dt = dt.combine(file_start.date(), time(hour=0, minute=0))
    expected_end_dt = dt.combine(file_end.date(), time(hour=23, minute=50))

    date_range_ok = (start_dt.date() == file_start.date()) and (end_dt.date() == file_end.date())

    intermediate, total_deltas = find_incorrect_data(file_path)
    total_deltas = total_deltas.to_pytimedelta()

    if not (start_ok and end_ok and date_range_ok) or total_deltas:
        missing_at_start = (start_dt - expected_start_dt).total_seconds() / 60
        missing_intermediate = total_deltas.total_seconds() / 60
        missing_at_end = (expected_end_dt - end_dt).total_seconds() / 60
        
        length_minutes = (end_dt - start_dt - total_deltas).total_seconds() / 60
        expected_length_minutes = ((file_end - file_start).total_seconds() / 60) + 24 * 60 - 10
        missing_minutes = expected_length_minutes - length_minutes
        wrong_minutes = abs(missing_at_start) + abs(missing_at_end) + missing_intermediate

        return {f: {
            "start_in_file": start_dt.strftime("%Y%m%d%H%M"),
            "end_in_file": end_dt.strftime("%Y%m%d%H%M"),
            "expected_start": file_start.strftime("%Y%m%d0000"),
            "expected_end": file_end.strftime("%Y%m%d2350"),
            "start_ok": start_ok,
            "end_ok": end_ok,
            "date_range_ok": date_range_ok,
            "length_minutes": length_minutes,
            "expected_length_minutes": expected_length_minutes,
            "missing_start": missing_at_start,
            "missing_end": missing_at_end,
            "missing_minutes": missing_minutes,
            "wrong_minutes": wrong_minutes,
            "intermediate_info": intermediate,
            "missing_intermediate": missing_intermediate,
            "metadata": {
                "folder": fold,
                "station_id": station_id
            }}
        }
    else:
        return {}

def check_content(fold):
    incorrect = {}
    unzip_folder = f"data/{fold}/unzips"
    print(f"{fold} has {len(os.listdir(unzip_folder))} files")

    num_workers = min(32, cpu_count())
    jobs = [(f, unzip_folder) for f in os.listdir(unzip_folder)]
    with Pool(num_workers) as pool:
        results = pool.starmap(process_file_check, jobs)

    for res in results:
        incorrect.update(res)

    print(f"{len(incorrect)} file(s) with incorrect values for {fold}!")
    return incorrect

def plot_incorrect(incorrect):
    fold_names = list(set([info["metadata"]["folder"] for info in incorrect.values()]))
    file_pct, time_pct = [], []

    for fold in fold_names:
        df = create_df(fold)
        total_files = len(df)
        fold_files = [v for v in incorrect.values() if v["metadata"]["folder"]==fold]
        file_pct.append(len(fold_files)/total_files*100 if total_files>0 else 0)
        total_seconds_all = sum((group["end_date"].max() - group["start_date"].min()).total_seconds()
                                for _, group in df.groupby("station_id"))

        wrong_seconds = sum([v["wrong_minutes"]*60 for v in fold_files])
        time_pct.append(wrong_seconds / total_seconds_all * 100 if total_seconds_all>0 else 0)

    plt.figure(figsize=(10,5))
    plt.bar(fold_names, file_pct, color='steelblue')
    # plt.title('Proportion of Files with Incorrect Temporal Alignment', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Files with Incorrect Content (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/incorrect_files_pct.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.bar(fold_names, time_pct, color='darkorange')
    # plt.title('Temporal Misalignment: Incorrect Time Relative to Total Station Time', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Incorrect Time (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/incorrect_time_pct.png", dpi=300)
    plt.close()

def resolve_incorrect(file_path, incorrect):
    with open(file_path, "r") as f:
        lines = f.readlines()

    header = lines[0]
    data_lines = lines[1:]

    cols = header.rstrip("\n").split(";")
    col_widths = [len(col) for col in cols]
    
    station_id = incorrect["metadata"]["station_id"]
    
    n_lines_start = int(incorrect["missing_start"] / 10)
    n_lines_end = int(incorrect["missing_end"] / 10)
    
    if n_lines_start < 0:
        data_lines = data_lines[-n_lines_start:]
    
    if n_lines_end < 0:
        data_lines = data_lines[:n_lines_end]

    intermediate_info = incorrect["intermediate_info"]
    if not intermediate_info.empty:
        gaps_list = list(intermediate_info.iterrows())
        for index, gap in reversed(gaps_list):
            gap_start_dt = gap["end_time"] - gap["delta"]
            gap_end_dt = gap["end_time"]
            lines_to_insert = int(gap["delta"].total_seconds() / 600) - 1
            
            end_time_str_check = gap_end_dt.strftime("%Y%m%d%H%M")
            insert_index = -1
            
            for idx, line in enumerate(data_lines):
                line_time_str = line.split(";")[1].strip() 
                if line_time_str == end_time_str_check:
                    insert_index = idx
                    break

            if insert_index != -1:
                new_lines = []
                current_dt = gap_start_dt
                
                for _ in range(lines_to_insert):
                    current_dt += timedelta(minutes=10)
                    new_time_str = current_dt.strftime("%Y%m%d%H%M")
                    
                    line_fields = [station_id.rjust(col_widths[0]), new_time_str.rjust(col_widths[1])]
                    for w in col_widths[2:]:
                        line_fields.append("-999".rjust(w))
                        
                    new_line = ";".join(line_fields) + "\n"
                    new_lines.append(new_line)
                
                data_lines[insert_index:insert_index] = new_lines

    if n_lines_start > 0:
        if not data_lines and header: # Should only occur when file is empty to beginn with
            start_dt = dt.strptime(incorrect["expected_start"], "%Y%m%d%H%M")
        else:
            start_dt = dt.strptime(incorrect["start_in_file"], "%Y%m%d%H%M")
        
        new_data_lines = []
        
        for i in range(n_lines_start, 0, -1):
            new_dt = start_dt - timedelta(minutes=10 * i)
            new_time_str = new_dt.strftime("%Y%m%d%H%M")
            
            line_fields = [station_id.rjust(col_widths[0]), new_time_str.rjust(col_widths[1])]
            for w in col_widths[2:]:
                line_fields.append("-999".rjust(w))
                
            new_line = ";".join(line_fields) + "\n"
            new_data_lines.append(new_line)

        data_lines = new_data_lines + data_lines

    if n_lines_end > 0:
        if not data_lines and header: # Should only occur when file is empty to beginn with
            end_dt = dt.strptime(incorrect["expected_start"], "%Y%m%d%H%M") 
        else:
            end_dt = dt.strptime(incorrect["end_in_file"], "%Y%m%d%H%M")
        
        for i in range(1, n_lines_end + 1):
            new_dt = end_dt + timedelta(minutes=10 * i)
            new_time_str = new_dt.strftime("%Y%m%d%H%M")
            
            line_fields = [station_id.rjust(col_widths[0]), new_time_str.rjust(col_widths[1])]
            for w in col_widths[2:]:
                line_fields.append("-999".rjust(w))
                
            new_line = ";".join(line_fields) + "\n"
            data_lines.append(new_line)

    with open(file_path, "w") as f:
        f.write(header)
        f.writelines(data_lines)

    print(f"Fixed {file_path}: Corrected {len(data_lines)} data lines. Total error was {incorrect['wrong_minutes']:.2f} minutes.")

    return incorrect["wrong_minutes"]

def resolve_incorrect_wrapper(args):
    file_path, incorrect_data = args
    return resolve_incorrect(file_path, incorrect_data)

def process_incorrect(incorrect):
    total_files, total_minutes = 0, 0

    num_workers = min(32, cpu_count())
    jobs = [
        (f"data/{values['metadata']['folder']}/unzips/{filename}", values)
        for filename, values in incorrect.items()
    ]
    with Pool(num_workers) as pool:
        results = pool.map(resolve_incorrect_wrapper, jobs)

    total_minutes = sum(results)
    total_files = len(results)

    print(f"Fixed {total_minutes} minutes in {total_files} files")

incorrect = {}
print("starting incorrect check")
for fold in folds:
    incorrect.update(check_content(fold))
    print(f"finished {fold}")

print("starting incorrect resolve")
process_incorrect(incorrect)
print("finished incorrect processing")

plot_incorrect(incorrect)


# concat files from same station

def concat_station(station_id_files, unzip_folder, concatenated_folder):
    station_id, files = station_id_files
    files_sorted = sorted(files, key=lambda x: x.split("_")[-3])
    output_file = f"{concatenated_folder}/{station_id}.txt"

    with open(output_file, "w") as out_f:
        for i, f in enumerate(files_sorted):
            with open(f"{unzip_folder}/{f}", "r") as in_f:
                lines = in_f.readlines()
                if i == 0:
                    out_f.write(lines[0])

                out_f.writelines(lines[1:])

def concat_station_files(fold):
    unzip_folder = f"data/{fold}/unzips"
    concatenated_folder = f"data/{fold}/concatenated"
    os.makedirs(concatenated_folder, exist_ok=True)

    station_files = {}
    for f in os.listdir(unzip_folder):
        if f.startswith(".") or not os.path.isfile(f"{unzip_folder}/{f}"):
            continue

        parts = f.split("_")
        station_id = parts[-1].split(".")[0]
        station_files.setdefault(station_id, []).append(f)

    station_items = list(station_files.items())
    workers = min(len(station_items), int(cpu_count() * 0.9))
    with Pool(workers) as pool:
        pool.starmap(concat_station, [(item, unzip_folder, concatenated_folder) for item in station_items])

    concat_counts = len(station_files.keys())
    print(f"{fold}: concatenated {concat_counts} files")
    return concat_counts

def plot_concatenations(concat_stats):
    fold_names = list(concat_stats.keys())
    concat_pct = []

    for fold in fold_names:
        total_files = len(create_df(fold))
        concat_pct.append(concat_stats[fold] / total_files * 100 if total_files > 0 else 0)

    plt.figure(figsize=(10, 5))
    plt.bar(fold_names, concat_pct, color='green')
    # plt.title('File Consolidation: Proportion of Multi-File Stations', fontsize=14, fontweight='bold')
    plt.xlabel('Variable Category', fontsize=48)
    plt.ylabel('Stations Requiring Concatenation (%)', fontsize=48)
    plt.tick_params(axis='both', which='major', labelsize=48)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("plots/v1/concatenated_files_pct.png", dpi=300)
    plt.close()

print("starting concatenation of station files...")
concat_stats = {}
for fold in folds:
    count = concat_station_files(fold)
    concat_stats[fold] = count

plot_concatenations(concat_stats)
