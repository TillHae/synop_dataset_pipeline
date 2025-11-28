import os
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def interpolate_variable(data: np.ndarray, max_gap: int = 6) -> tuple:
    """
    Attempt to interpolate NaN values along the time dimension.
    Only interpolates gaps smaller than max_gap to avoid unrealistic interpolation.
    """
    data_interp = data.copy()
    stats = {
        'total_nans': int(np.isnan(data).sum()),
        'interpolated': 0,
        'remaining_nans': 0,
        'gaps_too_large': 0
    }
    
    if stats['total_nans'] == 0:
        return data_interp, stats
    
    for station_idx in range(data.shape[0]):
        station_data = data[station_idx, :]
        if np.all(np.isnan(station_data)):
            continue
        
        series = pd.Series(station_data)
        is_nan = series.isna()
        nan_groups = []
        in_gap = False
        gap_start = None
        for i, is_missing in enumerate(is_nan):
            if is_missing and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_missing and in_gap:
                nan_groups.append((gap_start, i))
                in_gap = False
        
        if in_gap:
            nan_groups.append((gap_start, len(series)))
        
        for gap_start, gap_end in nan_groups:
            gap_size = gap_end - gap_start
            
            if gap_size <= max_gap:
                has_left = gap_start > 0 and not np.isnan(series.iloc[gap_start - 1])
                has_right = gap_end < len(series) and not np.isnan(series.iloc[gap_end])
                if has_left and has_right:
                    series.iloc[gap_start:gap_end] = series.iloc[gap_start:gap_end].interpolate(method='linear')
                    stats['interpolated'] += gap_size
                else:
                    stats['gaps_too_large'] += gap_size

            else:
                stats['gaps_too_large'] += gap_size
        
        data_interp[station_idx, :] = series.values
    
    stats['remaining_nans'] = int(np.isnan(data_interp).sum())
    return data_interp, stats


def extrapolate_variable(data: np.ndarray, max_extrap: int = 2) -> tuple:
    """
    Extrapolate NaN values using linear extrapolation.
    Handles three cases:
    1. Leading NaNs (at start of time series) - backward extrapolation
    2. Trailing NaNs (at end of time series) - forward extrapolation
    3. Edges of remaining internal gaps - backward (left edge) and forward (right edge)
    """
    data_extrap = data.copy()
    stats = {
        'total_nans': int(np.isnan(data).sum()),
        'extrapolated_forward': 0,
        'extrapolated_backward': 0,
        'remaining_nans': 0
    }
    
    if stats['total_nans'] == 0:
        return data_extrap, stats
    
    for station_idx in range(data.shape[0]):
        station_data = data[station_idx, :]
        if np.all(np.isnan(station_data)):
            continue
        
        series = pd.Series(station_data)
        valid_indices = series.notna()
        if not valid_indices.any():
            continue
        
        first_valid = valid_indices.idxmax()
        last_valid = len(series) - 1 - valid_indices[::-1].idxmax()
        if first_valid > 0:
            leading_nans = first_valid
            if leading_nans <= max_extrap and first_valid + 1 < len(series):
                # Use first 2-4 valid points to establish trend
                n_points = min(4, len(series) - first_valid)
                if n_points >= 2:
                    x_vals = np.arange(first_valid, first_valid + n_points)
                    y_vals = series.iloc[first_valid:first_valid + n_points].values
                    
                    slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                    intercept = y_vals[0] - slope * x_vals[0]
                    for i in range(first_valid):
                        series.iloc[i] = slope * i + intercept
                    
                    stats['extrapolated_backward'] += leading_nans
        
        if last_valid < len(series) - 1:
            trailing_nans = len(series) - 1 - last_valid
            if trailing_nans <= max_extrap and last_valid >= 1:
                # Use last 2-4 valid points to establish trend
                n_points = min(4, last_valid + 1)
                if n_points >= 2:
                    x_vals = np.arange(last_valid - n_points + 1, last_valid + 1)
                    y_vals = series.iloc[last_valid - n_points + 1:last_valid + 1].values
                    
                    slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                    intercept = y_vals[-1] - slope * x_vals[-1]
                    for i in range(last_valid + 1, len(series)):
                        series.iloc[i] = slope * i + intercept
                    
                    stats['extrapolated_forward'] += trailing_nans
        
        is_nan = series.isna()
        nan_groups = []
        in_gap = False
        gap_start = None
        
        for i, is_missing in enumerate(is_nan):
            if is_missing and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_missing and in_gap:
                nan_groups.append((gap_start, i))
                in_gap = False
        
        if in_gap:
            nan_groups.append((gap_start, len(series)))
        
        for gap_start, gap_end in nan_groups:
            gap_size = gap_end - gap_start
            has_left = gap_start > 0 and not np.isnan(series.iloc[gap_start - 1])
            has_right = gap_end < len(series) and not np.isnan(series.iloc[gap_end])
            
            if has_left:
                n_points = min(4, gap_start)
                if n_points >= 2:
                    x_vals = np.arange(gap_start - n_points, gap_start)
                    y_vals = series.iloc[gap_start - n_points:gap_start].values
                    
                    slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                    intercept = y_vals[-1] - slope * x_vals[-1]
                    fill_count = min(max_extrap, gap_size)
                    filled = 0
                    for i in range(gap_start, gap_start + fill_count):
                        if np.isnan(series.iloc[i]):
                            series.iloc[i] = slope * i + intercept
                            filled += 1
                    stats['extrapolated_backward'] += filled
            
            if has_right:
                n_points = min(4, len(series) - gap_end)
                if n_points >= 2:
                    x_vals = np.arange(gap_end, gap_end + n_points)
                    y_vals = series.iloc[gap_end:gap_end + n_points].values
                    
                    slope = (y_vals[-1] - y_vals[0]) / (x_vals[-1] - x_vals[0])
                    intercept = y_vals[0] - slope * x_vals[0]
                    fill_count = min(max_extrap, gap_size)
                    filled = 0
                    for i in range(gap_end - fill_count, gap_end):
                        if np.isnan(series.iloc[i]):
                            series.iloc[i] = slope * i + intercept
                            filled += 1
                    stats['extrapolated_forward'] += filled
        
        data_extrap[station_idx, :] = series.values
    
    stats['remaining_nans'] = int(np.isnan(data_extrap).sum())
    return data_extrap, stats


def process_file(nc_file: Path, output_dir: Path, max_gap: int = 6, max_extrap: int = 2) -> dict:
    print(f"\nProcessing: {nc_file.name}")
    
    try:
        ds = xr.open_dataset(nc_file)
        ds_processed = ds.copy()
        file_stats = {}
        total_interpolated = 0
        total_extrapolated = 0
        total_nans_before = 0
        total_nans_after = 0
        
        for var_name in ds.data_vars:
            if not np.issubdtype(ds[var_name].dtype, np.number):
                continue
            
            # Skip metadata variables - they should not be interpolated/extrapolated
            metadata_vars = ['height', 'latitude', 'longitude', 'QN']
            if var_name in metadata_vars:
                continue
            
            data = ds[var_name].values
            original_shape = data.shape
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            data_interp, interp_stats = interpolate_variable(data, max_gap)
            data_final, extrap_stats = extrapolate_variable(data_interp, max_extrap)
            
            data_final = data_final.reshape(original_shape)
            ds_processed[var_name].values = data_final
            combined_stats = {
                'total_nans': interp_stats['total_nans'],
                'interpolated': interp_stats['interpolated'],
                'extrapolated_forward': extrap_stats['extrapolated_forward'],
                'extrapolated_backward': extrap_stats['extrapolated_backward'],
                'remaining_nans': extrap_stats['remaining_nans']
            }
            
            file_stats[var_name] = combined_stats
            total_interpolated += interp_stats['interpolated']
            total_extrapolated += extrap_stats['extrapolated_forward'] + extrap_stats['extrapolated_backward']
            total_nans_before += interp_stats['total_nans']
            total_nans_after += extrap_stats['remaining_nans']
        
        output_file = output_dir / nc_file.name
        ds_processed.to_netcdf(output_file)
        
        ds.close()
        
        print(f"  NaNs before: {total_nans_before:,}")
        print(f"  Interpolated: {total_interpolated:,}")
        print(f"  Extrapolated: {total_extrapolated:,}")
        print(f"  NaNs after: {total_nans_after:,}")
        
        if total_nans_before > 0:
            total_filled = total_interpolated + total_extrapolated
            fill_pct = (total_filled / total_nans_before) * 100
            print(f"  Total fill success: {fill_pct:.1f}%")
        
        print(f"  Saved to: {output_file}")
        return {
            'filename': nc_file.name,
            'var_stats': file_stats,
            'total_nans_before': total_nans_before,
            'total_interpolated': total_interpolated,
            'total_extrapolated': total_extrapolated,
            'total_nans_after': total_nans_after
        }
        
    except Exception as e:
        print(f"  Error processing {nc_file.name}: {e}")
        return {
            'filename': nc_file.name,
            'error': str(e)
        }


def plot_interpolation_summary(all_results: list, output_dir: str = "plots/v4"):
    os.makedirs(output_dir, exist_ok=True)
    var_stats_global = {}
    for result in all_results:
        if 'error' in result:
            continue
        
        for var_name, stats in result.get('var_stats', {}).items():
            if var_name not in var_stats_global:
                var_stats_global[var_name] = {
                    'total_nans': 0,
                    'interpolated': 0,
                    'extrapolated': 0,
                    'remaining_nans': 0
                }
            
            var_stats_global[var_name]['total_nans'] += stats['total_nans']
            var_stats_global[var_name]['interpolated'] += stats['interpolated']
            var_stats_global[var_name]['extrapolated'] += stats.get('extrapolated_forward', 0) + stats.get('extrapolated_backward', 0)
            var_stats_global[var_name]['remaining_nans'] += stats['remaining_nans']
    
    if not var_stats_global:
        return
    
    fill_success = {}
    for var_name, stats in var_stats_global.items():
        if stats['total_nans'] > 0:
            total_filled = stats['interpolated'] + stats['extrapolated']
            success_rate = (total_filled / stats['total_nans']) * 100
            fill_success[var_name] = success_rate
    
    plt.figure(figsize=(12, 6))
    vars_sorted = sorted(fill_success.keys(), key=lambda x: fill_success[x], reverse=True)
    values = [fill_success[v] for v in vars_sorted]
    colors = ['#1a9850' if v > 50 else '#fee08b' if v > 10 else '#d73027' for v in values]
    
    plt.bar(range(len(vars_sorted)), values, color=colors)
    plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
    plt.xlabel('Variable', fontsize=12)
    plt.ylabel('Fill Success Rate (%)', fontsize=12)
    plt.title('Percentage of NaN Values Filled (Interpolation + Extrapolation)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fill_success.png", dpi=300)
    plt.close()
    print(f"\nSaved fill success plot to {output_dir}/fill_success.png")

    nans_before = [var_stats_global[v]['total_nans'] for v in vars_sorted]
    nans_after = [var_stats_global[v]['remaining_nans'] for v in vars_sorted]
    plt.figure(figsize=(12, 6))
    x = np.arange(len(vars_sorted))
    width = 0.35
    
    plt.bar(x - width/2, nans_before, width, label='Before', color='#d73027', alpha=0.7)
    plt.bar(x + width/2, nans_after, width, label='After', color='#1a9850', alpha=0.7)
    plt.xlabel('Variable', fontsize=12)
    plt.ylabel('Number of NaN Values', fontsize=12)
    plt.title('NaN Count Before and After Gap Filling', fontsize=14, fontweight='bold')
    plt.xticks(x, vars_sorted, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/nan_reduction.png", dpi=300)
    plt.close()
    print(f"Saved NaN reduction plot to {output_dir}/nan_reduction.png")
    
    interp_vals = [var_stats_global[v]['interpolated'] for v in vars_sorted]
    extrap_vals = [var_stats_global[v]['extrapolated'] for v in vars_sorted]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(vars_sorted))
    width = 0.35
    
    plt.bar(x - width/2, interp_vals, width, label='Interpolated', color='#4575b4', alpha=0.8)
    plt.bar(x + width/2, extrap_vals, width, label='Extrapolated', color='#d73027', alpha=0.8)
    plt.xlabel('Variable', fontsize=12)
    plt.ylabel('Number of Values Filled', fontsize=12)
    plt.title('Gap Filling Method Breakdown by Variable', fontsize=14, fontweight='bold')
    plt.xticks(x, vars_sorted, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_breakdown.png", dpi=300)
    plt.close()
    print(f"Saved method breakdown plot to {output_dir}/method_breakdown.png")


def save_interpolation_report(all_results: list, output_path: str = "gap_filling_report.txt"):
    with open(output_path, 'w') as f:
        f.write("GAP FILLING REPORT (INTERPOLATION + EXTRAPOLATION)\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            if 'error' in result:
                f.write(f"\nFILE: {result['filename']}\n")
                f.write(f"ERROR: {result['error']}\n")
                continue
            
            f.write(f"\nFILE: {result['filename']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total NaNs before: {result['total_nans_before']:,}\n")
            f.write(f"Interpolated: {result['total_interpolated']:,}\n")
            f.write(f"Extrapolated: {result['total_extrapolated']:,}\n")
            f.write(f"Total NaNs after: {result['total_nans_after']:,}\n")
            
            if result['total_nans_before'] > 0:
                total_filled = result['total_interpolated'] + result['total_extrapolated']
                success_rate = (total_filled / result['total_nans_before']) * 100
                f.write(f"Fill success rate: {success_rate:.2f}%\n")
            
            f.write("\nPer-variable statistics:\n")
            for var_name, stats in result.get('var_stats', {}).items():
                if stats['total_nans'] > 0:
                    total_filled = stats['interpolated'] + stats.get('extrapolated_forward', 0) + stats.get('extrapolated_backward', 0)
                    var_success = (total_filled / stats['total_nans']) * 100
                    f.write(f"  {var_name}:\n")
                    f.write(f"    Total NaNs: {stats['total_nans']:,}\n")
                    f.write(f"    Interpolated: {stats['interpolated']:,}\n")
                    f.write(f"    Extrapolated (forward): {stats.get('extrapolated_forward', 0):,}\n")
                    f.write(f"    Extrapolated (backward): {stats.get('extrapolated_backward', 0):,}\n")
                    f.write(f"    Remaining: {stats['remaining_nans']:,}\n")
                    f.write(f"    Fill rate: {var_success:.1f}%\n")
    
    print(f"\nGap filling report saved to: {output_path}")


def main():
    print("Creating Dataset v4: Gap Filling (Interpolation + Extrapolation)")
    print("="*80)
    
    input_dir = Path("data/datasetv3")
    output_dir = Path("data/datasetv4")
    max_intrap = 6
    max_extrap = 2
    if not input_dir.exists():
        print(f"\nError: Input directory '{input_dir}' does not exist.")
        print("   Please run create_datasetv3.py first.")
        return
    
    nc_files = sorted(input_dir.glob('*.nc'))
    if not nc_files:
        print(f"\nError: No NetCDF files found in '{input_dir}'")
        return
    
    print(f"\nFound {len(nc_files)} NetCDF file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max gap for interpolation: {max_intrap} timesteps (1 hour)")
    print(f"Max gap for extrapolation: {max_extrap} timesteps (20 minutes)")
    print("\nNote: Interpolation requires valid neighbors; extrapolation fills leading/trailing gaps")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for nc_file in nc_files:
        result = process_file(nc_file, output_dir, max_intrap, max_extrap)
        all_results.append(result)
    
    plot_interpolation_summary(all_results)
    save_interpolation_report(all_results)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful_files = [r for r in all_results if 'error' not in r]
    if successful_files:
        total_nans_before = sum(r['total_nans_before'] for r in successful_files)
        total_interpolated = sum(r['total_interpolated'] for r in successful_files)
        total_extrapolated = sum(r['total_extrapolated'] for r in successful_files)
        total_nans_after = sum(r['total_nans_after'] for r in successful_files)
        
        print(f"\nTotal NaN values across all files:")
        print(f"  Before gap filling: {total_nans_before:,}")
        print(f"  Interpolated: {total_interpolated:,}")
        print(f"  Extrapolated: {total_extrapolated:,}")
        print(f"  Remaining NaN values: {total_nans_after:,}")
        
        if total_nans_before > 0:
            total_filled = total_interpolated + total_extrapolated
            overall_success = (total_filled / total_nans_before) * 100
            print(f"\nOverall gap filling success rate: {overall_success:.2f}%")
            interp_pct = (total_interpolated / total_filled * 100) if total_filled > 0 else 0
            extrap_pct = (total_extrapolated / total_filled * 100) if total_filled > 0 else 0
            print(f"  Interpolation: {interp_pct:.1f}% | Extrapolation: {extrap_pct:.1f}%")
            if overall_success < 10:
                print("\nNote: Low fill rate is expected due to sparse data")
                print("      Most NaN values lack neighboring valid values for gap filling")
    
    print(f"\nDataset v4 created successfully!")
    print(f"   Data saved to: {output_dir}/")


if __name__ == "__main__":
    main()