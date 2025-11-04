from utils import create_df
import matplotlib.pyplot as plt

folds = ["air_temp", "ex_temp", "ex_wind", "precip", "solar", "wind"]

def station_duration(df):
    # Compute duration per file in days
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days
    
    # Sum durations per station
    durations = df.groupby("station_id")["duration_days"].sum().reset_index()
    
    # Convert total duration to integer years
    durations["duration_years"] = (durations["duration_days"] / 365.25).astype(int)
    
    # Count how many stations have each total duration
    year_counts = durations["duration_years"].value_counts().sort_index()
    return year_counts

def plot_station_duration(total_year_counts):
    plt.figure(figsize=(10, 6))
    
    for fold, year_counts in total_year_counts.items():
        plt.bar(year_counts.index, year_counts.values, alpha=0.6, label=fold)
    
    plt.xlabel("Total Duration (years)")
    plt.ylabel("Number of Stations")
    plt.title("Number of Stations by Total Duration per Fold")
    plt.legend()
    plt.xticks(range(0, max(max(yc.index) for yc in total_year_counts.values()) + 1))
    plt.tight_layout()
    plt.savefig("plots/station_durations.png")
    plt.close()

print("--- Create Station Duration Plot ---")

year_counts = {}
for fold in folds:
    df = create_df(fold)
    year_counts[fold] = station_duration(df)

plot_station_duration(year_counts)