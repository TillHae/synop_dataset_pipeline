from utils import create_df
import matplotlib.pyplot as plt
import pandas as pd

folds = ["air_temp", "ex_temp", "ex_wind", "precip", "solar", "wind"]

def station_duration(df):
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    durations = df.groupby("station_id")["duration_days"].sum().reset_index()
    durations["duration_years"] = (durations["duration_days"] / 365.25).astype(int)
    
    year_counts = durations["duration_years"].value_counts().sort_index()
    return year_counts

def plot_station_duration(total_year_counts):
    plt.figure(figsize=(10, 6))
    bottoms = None 
    sorted_keys = sorted(total_year_counts.keys())
    
    for fold in sorted_keys:
        year_counts = total_year_counts[fold]
        if bottoms is None:
            full_index = pd.Series(0, index=range(0, 35))
            bottoms = full_index
        
        current_data = full_index.add(year_counts, fill_value=0).sub(full_index, fill_value=0)
        plt.bar(current_data.index, current_data.values, alpha=0.8, 
                label=fold, bottom=bottoms.loc[current_data.index])
        
        bottoms = bottoms.add(current_data, fill_value=0) 

    plt.xlabel("Total Duration (years)")
    plt.ylabel("Number of Stations")
    plt.title("Number of Stations by Total Duration per Fold")
    plt.legend()
    max_duration = max(max(yc.index) for yc in total_year_counts.values()) 
    plt.xticks(range(0, max_duration + 1))
    plt.tight_layout()
    plt.savefig("plots/station_durations.png")
    plt.close()

print("--- Create Station Duration Plot ---")

year_counts = {}
for fold in folds:
    df = create_df(fold)
    year_counts[fold] = station_duration(df)

plot_station_duration(year_counts)