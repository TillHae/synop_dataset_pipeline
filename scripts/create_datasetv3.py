import os
from glob import glob
from pathlib import Path

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def calculate_global_nan_percentages(input_dir: Path) -> dict:
    print("\nCalculating global NaN percentages across all years...")
    nc_files = sorted(input_dir.glob('*.nc'))
    if not nc_files:
        raise ValueError(f"No NetCDF files found in {input_dir}")
    
    global_stats = {}
    for nc_file in nc_files:
        print(f"  Reading {nc_file.name}...")
        ds = xr.open_dataset(nc_file)
        for var_name in ds.data_vars:
            if not np.issubdtype(ds[var_name].dtype, np.number):
                continue
            
            if var_name not in global_stats:
                global_stats[var_name] = {'total': 0, 'nan': 0}
            
            total_values = ds[var_name].size
            nan_count = int(np.isnan(ds[var_name]).sum())
            global_stats[var_name]['total'] += total_values
            global_stats[var_name]['nan'] += nan_count
        
        ds.close()
    
    global_nan_percentages = {}
    for var_name, stats in global_stats.items():
        if stats['total'] > 0:
            nan_pct = (stats['nan'] / stats['total']) * 100
            global_nan_percentages[var_name] = nan_pct
    
    return global_nan_percentages


def identify_variables_to_remove(global_nan_percentages: dict, threshold: float = 90.0) -> tuple:
    vars_to_keep = []
    vars_to_remove = {}
    for var_name, nan_pct in global_nan_percentages.items():
        if nan_pct <= threshold:
            vars_to_keep.append(var_name)
        else:
            vars_to_remove[var_name] = nan_pct
    
    return vars_to_keep, vars_to_remove


def process_file(nc_file: Path, output_dir: Path, vars_to_keep: list) -> dict:
    print(f"\nProcessing: {nc_file.name}")
    try:
        ds = xr.open_dataset(nc_file)
        original_var_count = len(ds.data_vars)
        vars_in_file = [v for v in vars_to_keep if v in ds.data_vars]
        ds_filtered = ds[vars_in_file]
        remaining_var_count = len(ds_filtered.data_vars)
        removed_count = original_var_count - remaining_var_count
        
        print(f"  Original variables: {original_var_count}")
        print(f"  Remaining variables: {remaining_var_count}")
        print(f"  Removed: {removed_count}")
        
        output_file = output_dir / nc_file.name
        ds_filtered.to_netcdf(output_file)
        print(f"  Saved to: {output_file}")
        
        ds.close()
        return {
            'filename': nc_file.name,
            'original_vars': original_var_count,
            'remaining_vars': remaining_var_count
        }
        
    except Exception as e:
        print(f"  Error processing {nc_file.name}: {e}")
        return {
            'filename': nc_file.name,
            'error': str(e)
        }


def plot_removal_summary(global_nan_percentages: dict, vars_to_remove: dict, 
                        threshold: float, output_dir: str = "plots/v3"):
    os.makedirs(output_dir, exist_ok=True)
    if not vars_to_remove:
        print("\nNo variables were removed (all below 90% NaN threshold)")
        return
    
    plt.figure(figsize=(12, 6))
    vars_sorted = sorted(vars_to_remove.keys(), key=lambda x: vars_to_remove[x], reverse=True)
    values = [vars_to_remove[v] for v in vars_sorted]
    
    plt.barh(range(len(vars_sorted)), values, color='#d73027')
    plt.yticks(range(len(vars_sorted)), vars_sorted)
    plt.xlabel('Global NaN Percentage (%)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.title(f'Variables Removed Due to High Missing Data (>{threshold}% NaN Globally)', fontsize=14, fontweight='bold')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'{threshold}% threshold')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/removed_variables.png", dpi=300)
    plt.close()
    print(f"\nSaved removal summary plot to {output_dir}/removed_variables.png")
    
    plt.figure(figsize=(14, 8))
    all_vars_sorted = sorted(global_nan_percentages.keys(), key=lambda x: global_nan_percentages[x], reverse=True)
    all_values = [global_nan_percentages[v] for v in all_vars_sorted]
    colors = ['#d73027' if v > threshold else '#1a9850' for v in all_values]
    
    plt.barh(range(len(all_vars_sorted)), all_values, color=colors)
    plt.yticks(range(len(all_vars_sorted)), all_vars_sorted, fontsize=8)
    plt.xlabel('Global NaN Percentage (%)', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.title('Global NaN Percentage for All Variables (Red = Removed, Green = Kept)', fontsize=14, fontweight='bold')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'{threshold}% threshold')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_variables_nan.png", dpi=300)
    plt.close()
    print(f"Saved global NaN plot to {output_dir}/all_variables_nan.png")


def main():
    print("Creating Dataset v3: Removing High-NaN Variables (Global Analysis)")
    print("="*80)
    
    input_dir = Path("data/datasetv2")
    output_dir = Path("data/datasetv3")
    threshold = 90.0
    if not input_dir.exists():
        print(f"\nError: Input directory '{input_dir}' does not exist.")
        print("   Please run numerical_checks.py first to create datasetv2.")
        return
    
    nc_files = sorted(input_dir.glob('*.nc'))
    if not nc_files:
        print(f"\nError: No NetCDF files found in '{input_dir}'")
        return
    
    print(f"\nFound {len(nc_files)} NetCDF file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Global NaN threshold: {threshold}%")
    
    global_nan_percentages = calculate_global_nan_percentages(input_dir)
    vars_to_keep, vars_to_remove = identify_variables_to_remove(global_nan_percentages, threshold)
    
    print(f"\n{'='*80}")
    print("GLOBAL ANALYSIS RESULTS")
    print("="*80)
    print(f"\nTotal unique variables: {len(global_nan_percentages)}")
    print(f"Variables to keep: {len(vars_to_keep)}")
    print(f"Variables to remove: {len(vars_to_remove)}")
    
    if vars_to_remove:
        print(f"\nVariables being removed (>{threshold}% NaN globally):")
        for var_name, nan_pct in sorted(vars_to_remove.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {var_name}: {nan_pct:.2f}% NaN")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("PROCESSING FILES")
    print("="*80)
    
    all_results = []
    for nc_file in nc_files:
        result = process_file(nc_file, output_dir, vars_to_keep)
        all_results.append(result)
    
    plot_removal_summary(global_nan_percentages, vars_to_remove, threshold)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful_files = [r for r in all_results if 'error' not in r]
    if successful_files:
        final_var_count = successful_files[0]['remaining_vars']
        print(f"\nAll files now have {final_var_count} consistent variables")
        print(f"Removed {len(vars_to_remove)} variables globally")
    
    print(f"\nDataset v3 created successfully!")
    print(f"   All years have the same {len(vars_to_keep)} features")
    print(f"   Cleaned data saved to: {output_dir}/")


if __name__ == "__main__":
    main()