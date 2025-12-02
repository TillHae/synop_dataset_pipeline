import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class NumericalChecker:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.results = {}
        
        # Define metadata variables that should be excluded from most checks
        self.metadata_vars = ['height', 'latitude', 'longitude', 'QN', 'eor']
        
        # Define physically reasonable ranges for german meteorological variables
        self.variable_ranges = {
            # Temperature variables (in Celsius)
            # Germany: Record low ~-45°C (1929), Record high ~42°C (2019)
            'TU': (-50, 45),      # Air temperature (10-min average)
            'TX_10': (-50, 45),   # Max temperature in 10-min interval
            'TX5_10': (-50, 45),  # Max temperature at 5cm height
            'TN_10': (-50, 45),   # Min temperature in 10-min interval
            'TN5_10': (-50, 45),  # Min temperature at 5cm (ground can be colder)
            
            # Wind variables (m/s)
            # Germany: Typical max gusts ~40 m/s, extreme storms ~50 m/s
            'FF_10': (0, 45),     # Wind speed (10-min average)
            'FX_10': (0, 55),     # Max wind gust in 10-min interval
            'FNX_10': (0, 55),    # Max wind gust
            'FMX_10': (0, 55),    # Max wind gust
            
            # Wind direction (degrees)
            'DD_10': (0, 360),    # Wind direction
            'DX_10': (0, 360),    # Wind direction at max gust
            
            # Precipitation (mm)
            # Germany: Extreme events can reach 50-100mm in 10 minutes
            'RWS_10': (0, 100),   # Precipitation amount in 10-min interval
            'RWS_DAU_10': (0, 10), # Precipitation duration (seconds, max 10 min)
            'RWS_IND_10': (0, 3),   # Precipitation indicator (0=no, 1=yes, 2,3=older_versions with heating)
            
            # Solar radiation (J/cm²)
            # Germany latitude: ~47-55°N, moderate solar radiation
            'GS_10': (0, 500),    # Global radiation (10-min sum)
            'DS_10': (0, 350),    # Diffuse radiation (10-min sum)
            'SD_10': (0, 700),    # Sunshine duration (seconds, max 10 min)
            'LS_10': (0, 500),    # Long-wave radiation (10-min sum)

            # Metadata (coordinates)
            # Germany: Latitude 47-55°N, Longitude 6-15°E, Elevation -3.5m to 2962m
            'latitude': (46, 56),
            'longitude': (5, 16),
            'height': (-50, 3000),
        }
    
    def check_nan_inf(self, data: xr.DataArray, var_name: str) -> Dict:
        total_values = data.size
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        
        nan_percentage = (nan_count / total_values * 100) if total_values > 0 else 0
        inf_percentage = (inf_count / total_values * 100) if total_values > 0 else 0
        
        return {
            'total_values': total_values,
            'nan_count': nan_count,
            'nan_percentage': nan_percentage,
            'inf_count': inf_count,
            'inf_percentage': inf_percentage,
            'has_issues': nan_count > 0 or inf_count > 0
        }
    
    def check_range(self, data: xr.DataArray, var_name: str) -> Dict:
        if var_name not in self.variable_ranges:
            return {
                'range_defined': False,
                'message': f'No range defined for {var_name}'
            }
        
        min_val, max_val = self.variable_ranges[var_name]
        valid_data = data.values[~np.isnan(data.values)]
        
        if len(valid_data) == 0:
            return {
                'range_defined': True,
                'expected_range': (min_val, max_val),
                'all_nan': True
            }
        
        below_min = np.sum(valid_data < min_val)
        above_max = np.sum(valid_data > max_val)
        in_range = len(valid_data) - below_min - above_max
        
        return {
            'range_defined': True,
            'expected_range': (min_val, max_val),
            'actual_range': (float(np.min(valid_data)), float(np.max(valid_data))),
            'below_min': int(below_min),
            'above_max': int(above_max),
            'in_range': int(in_range),
            'in_range_percentage': (in_range / len(valid_data) * 100) if len(valid_data) > 0 else 0,
            'has_issues': below_min > 0 or above_max > 0
        }
    
    def check_stationarity(self, data: xr.DataArray, var_name: str) -> Dict:
        """Test inter-annual consistency for atmospheric data."""
        if 'time' not in data.dims:
            return {
                'stationarity_check': False,
                'message': 'No time dimension found'
            }
        
        if 'station_id' not in data.dims:
            return {
                'stationarity_check': False,
                'message': 'No station_id dimension - need per-station analysis'
            }
        
        try:
            times = pd.to_datetime(data.coords['time'].values)
            years = times.year.unique()
            if len(years) < 2:
                return {
                    'stationarity_check': False,
                    'message': 'Need at least 2 years of data for inter-annual comparison'
                }

        except Exception as e:
            return {
                'stationarity_check': False,
                'message': f'Error extracting years: {str(e)}'
            }
        
        station_year_stats = []
        n_stations = data.shape[0]
        for station_idx in range(n_stations):
            station_data = data.values[station_idx, :]
            station_times = pd.to_datetime(data.coords['time'].values)
            year_means = []
            year_stds = []
            for year in years:
                year_mask = station_times.year == year
                year_data = station_data[year_mask]
                valid_data = year_data[~np.isnan(year_data)]
                if len(valid_data) >= 30:
                    year_means.append(np.mean(valid_data))
                    year_stds.append(np.std(valid_data))
            
            if len(year_means) >= 2:
                station_year_stats.append({
                    'means': year_means,
                    'stds': year_stds
                })
        
        if len(station_year_stats) < 2:
            return {
                'stationarity_check': False,
                'message': 'Insufficient stations with multi-year data'
            }
        
        # Calculate coefficient of variation (CV) for means and stds across years
        # CV = std / mean - measures relative variability
        mean_cvs = []
        std_cvs = []
        for stats in station_year_stats:
            if len(stats['means']) >= 2:
                mean_of_means = np.mean(stats['means'])
                std_of_means = np.std(stats['means'])
                if mean_of_means != 0:
                    mean_cvs.append(std_of_means / abs(mean_of_means))
                
                mean_of_stds = np.mean(stats['stds'])
                std_of_stds = np.std(stats['stds'])
                if mean_of_stds != 0:
                    std_cvs.append(std_of_stds / mean_of_stds)
        
        if not mean_cvs:
            return {
                'stationarity_check': False,
                'message': 'Could not calculate inter-annual variability'
            }
        
        avg_mean_cv = np.mean(mean_cvs)
        avg_std_cv = np.mean(std_cvs)
        # Flag as issue if CV is very high (>0.3 = 30% variation year-to-year)
        has_issues = avg_mean_cv > 0.3 or avg_std_cv > 0.3
        return {
            'stationarity_check': True,
            'n_years': len(years),
            'n_stations_analyzed': len(station_year_stats),
            'mean_cv': float(avg_mean_cv),  # Coefficient of variation for yearly means
            'std_cv': float(avg_std_cv),    # Coefficient of variation for yearly stds
            'has_issues': has_issues,
            'interpretation': 'High CV indicates inconsistent statistics across years (sensor drift/climate trend)'
        }
    
    def check_variance_homogeneity(self, data: xr.DataArray, var_name: str) -> Dict:
        """Test for variance homogeneity across stations using Levene's test."""
        if 'station_id' not in data.dims:
            return {
                'variance_check': False,
                'message': 'No station_id dimension found'
            }
        
        station_data = []
        for i in range(data.shape[0]):
            station_vals = data.values[i, :]
            valid_vals = station_vals[~np.isnan(station_vals)]
            if len(valid_vals) >= 10:
                station_data.append(valid_vals)
        
        if len(station_data) < 2:
            return {
                'variance_check': False,
                'message': 'Insufficient stations with valid data'
            }
        
        try:
            # Levene's test (null hypothesis: equal variances)
            levene_statistic, levene_pvalue = stats.levene(*station_data)
            homogeneous = levene_pvalue > 0.05
            
            variances = [np.var(s) for s in station_data]
            
            return {
                'variance_check': True,
                'levene_statistic': float(levene_statistic),
                'levene_pvalue': float(levene_pvalue),
                'homogeneous_variance': homogeneous,
                'min_variance': float(np.min(variances)),
                'max_variance': float(np.max(variances)),
                'mean_variance': float(np.mean(variances)),
                'variance_ratio': float(np.max(variances) / np.min(variances)) if np.min(variances) > 0 else np.inf,
                'has_issues': not homogeneous
            }
        except Exception as e:
            return {
                'variance_check': False,
                'message': f'Error in variance test: {str(e)}'
            }
    
    def check_distribution(self, data: xr.DataArray, var_name: str) -> Dict:
        """Test distribution using Kolmogorov-Smirnov test for normality."""
        valid_data = data.values[~np.isnan(data.values)]
        
        if len(valid_data) < 20:
            return {
                'distribution_check': False,
                'message': 'Insufficient data for distribution tests'
            }
        
        try:
            # Kolmogorov-Smirnov test for normality
            ks_statistic, ks_pvalue = stats.kstest(valid_data, 'norm', args=(np.mean(valid_data), np.std(valid_data)))
            is_normal = ks_pvalue > 0.05
            
            # Calculate distribution statistics
            skewness = float(stats.skew(valid_data))
            kurtosis = float(stats.kurtosis(valid_data))
            
            return {
                'distribution_check': True,
                'ks_statistic': float(ks_statistic),
                'ks_pvalue': float(ks_pvalue),
                'is_normal': is_normal,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'has_issues': False
            }
        except Exception as e:
            return {
                'distribution_check': False,
                'message': f'Error in distribution test: {str(e)}'
            }
    
    def check_temporal_consistency(self, data: xr.DataArray, var_name: str) -> Dict:
        """Check for unrealistic temporal jumps based on variable-specific physical constraints."""
        if 'time' not in data.dims:
            return {
                'temporal_check': False,
                'message': 'No time dimension found'
            }
        
        # Define variable-specific maximum realistic changes per 10-minute interval
        max_realistic_changes = {
            # Temperature: Can't should not more than ~5°C in 10 minutes under normal conditions
            'TU': 5.0, 'TX_10': 5.0, 'TX5_10': 5.0, 'TN_10': 5.0, 'TN5_10': 5.0,
            'TT_10': 5.0, 'TM5_10': 5.0, 'TD_10': 5.0,
            
            # Wind: Can change rapidly (gusts), so those are generous threshold
            'FF_10': 30.0,
            'FX_10': 40.0, 'FNX_10': 40.0, 'FMX_10': 40.0,
            
            # Wind direction: Can change 360° instantly (wind shift), so skip this check
            
            # Precipitation: Highly volatile, can go 0→50mm in 10 min (cloudburst)
            # => Only flag if it exceeds physical maximum
            'RWS_10': 100.0,  # Use range maximum as threshold
            'RWS_DAU_10': 10.0,  # Duration can't exceed 10 minutes
            
            # Solar radiation: Changes dramatically (night→day), but gradually
            # Flag only impossible jumps (sensor errors)
            'GS_10': 500.0,  # Can go from 0 to max in one interval (clouds)
            'DS_10': 350.0,
            'SD_10': 600.0,
            'LS_10': 500.0,
            
            # Pressure: Very stable, shouldn't change much in 10 minutes
            'PP_10': 5.0,  # hPa - anything more is likely sensor error
            
            # Humidity: Can change rapidly but not instantaneously
            'RF_10': 30.0,  # % - rapid but not instant changes
        }
        
        if var_name not in max_realistic_changes:
            return {
                'temporal_check': False,
                'message': f'No temporal threshold defined for {var_name}'
            }
        
        threshold = max_realistic_changes[var_name]
        if len(data.dims) > 1:
            if 'station_id' in data.dims:
                all_large_jumps = []
                all_max_diffs = []
                for i in range(data.shape[0]):
                    station_data = data.values[i, :]
                    valid_mask = ~np.isnan(station_data)
                    if np.sum(valid_mask) < 2:
                        continue
                    
                    diffs = np.diff(station_data[valid_mask])
                    if len(diffs) > 0:
                        large_jumps = np.sum(np.abs(diffs) > threshold)
                        all_large_jumps.append(large_jumps)
                        all_max_diffs.append(np.max(np.abs(diffs)))
                
                if not all_large_jumps:
                    return {
                        'temporal_check': False,
                        'message': 'Insufficient valid data points'
                    }
                
                total_large_jumps = sum(all_large_jumps)
                max_diff = max(all_max_diffs) if all_max_diffs else 0
                return {
                    'temporal_check': True,
                    'threshold': float(threshold),
                    'max_diff': float(max_diff),
                    'large_jumps': int(total_large_jumps),
                    'has_issues': total_large_jumps > 0
                }
            else:
                data_1d = data

        else:
            data_1d = data
        
        valid_mask = ~np.isnan(data_1d.values)
        if np.sum(valid_mask) < 2:
            return {
                'temporal_check': False,
                'message': 'Insufficient valid data points'
            }
        
        diffs = np.diff(data_1d.values[valid_mask])
        if len(diffs) == 0:
            return {
                'temporal_check': False,
                'message': 'No consecutive valid values'
            }
        
        max_diff = np.max(np.abs(diffs))
        large_jumps = np.sum(np.abs(diffs) > threshold)
        return {
            'temporal_check': True,
            'threshold': float(threshold),
            'max_diff': float(max_diff),
            'large_jumps': int(large_jumps),
            'has_issues': large_jumps > 0
        }
    
    def check_variable(self, data: xr.DataArray, var_name: str) -> Dict:
        print(f"  Checking variable: {var_name}")
        checks = {
            'variable_name': var_name,
            'dtype': str(data.dtype),
            'shape': data.shape,
            'dimensions': list(data.dims),
            'is_metadata': var_name in self.metadata_vars
        }
        
        checks['nan_inf_check'] = self.check_nan_inf(data, var_name)
        checks['range_check'] = self.check_range(data, var_name)
        if var_name not in self.metadata_vars:
            checks['stationarity_check'] = self.check_stationarity(data, var_name)
            checks['variance_check'] = self.check_variance_homogeneity(data, var_name)
            checks['distribution_check'] = self.check_distribution(data, var_name)
            checks['temporal_check'] = self.check_temporal_consistency(data, var_name)
        else:
            checks['stationarity_check'] = {'stationarity_check': False, 'message': 'Skipped for metadata variable'}
            checks['variance_check'] = {'variance_check': False, 'message': 'Skipped for metadata variable'}
            checks['distribution_check'] = {'distribution_check': False, 'message': 'Skipped for metadata variable'}
            checks['temporal_check'] = {'temporal_check': False, 'message': 'Skipped for metadata variable'}
        
        has_any_issues = any([
            checks['nan_inf_check'].get('has_issues', False),
            checks['range_check'].get('has_issues', False),
            checks.get('stationarity_check', {}).get('has_issues', False),
            checks.get('variance_check', {}).get('has_issues', False),
            checks.get('temporal_check', {}).get('has_issues', False)
        ])

        checks['status'] = 'WARNING' if has_any_issues else 'OK'
        return checks
    
    def check_dataset(self, ds: xr.Dataset, filename: str = None) -> Dict:
        print(f"\nChecking dataset: {filename or 'unnamed'}")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Coordinates: {list(ds.coords)}")
        
        dataset_results = {
            'filename': filename,
            'dimensions': dict(ds.dims),
            'variables': {},
            'coordinates': {}
        }
        for var_name in ds.data_vars:
            dataset_results['variables'][var_name] = self.check_variable(
                ds[var_name], var_name
            )
        
        for coord_name in ds.coords:
            if coord_name not in ['time', 'station_id']:
                if np.issubdtype(ds[coord_name].dtype, np.number):
                    dataset_results['coordinates'][coord_name] = self.check_variable(
                        ds[coord_name], coord_name
                    )
        
        return dataset_results
    
    def run_checks(self) -> Dict:
        if self.dataset_path.is_file():
            files = [self.dataset_path]
        elif self.dataset_path.is_dir():
            files = sorted(self.dataset_path.glob('*.nc'))
        else:
            raise ValueError(f"Path does not exist: {self.dataset_path}")
        
        if not files:
            raise ValueError(f"No NetCDF files found in {self.dataset_path}")
        
        print(f"Found {len(files)} NetCDF file(s) to check")
        
        all_results = {}
        for nc_file in files:
            try:
                ds = xr.open_dataset(nc_file)
                results = self.check_dataset(ds, nc_file.name)
                all_results[nc_file.name] = results
                ds.close()
            except Exception as e:
                print(f"  ERROR processing {nc_file.name}: {e}")
                all_results[nc_file.name] = {
                    'error': str(e)
                }
        
        self.results = all_results
        return all_results
    
    def print_summary(self):
        if not self.results:
            print("No results available. Run check_dataset() first.")
            return
        
        print("\n" + "="*80)
        print("NUMERICAL CHECKS SUMMARY")
        print("="*80)
        
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                print(f"\n{filename}: ERROR - {file_results['error']}")
                continue
            
            print(f"\n {filename}")
            print(f"   Dimensions: {file_results['dimensions']}")
            
            total_vars = len(file_results['variables'])
            warning_vars = sum(
                1 for v in file_results['variables'].values() 
                if v.get('status') == 'WARNING'
            )
            
            print(f"   Variables: {total_vars} total, {warning_vars} with warnings")
            
            for var_name, var_results in file_results['variables'].items():
                if var_results.get('status') == 'WARNING':
                    print(f"\n    {var_name}:")
                    
                    if var_results['nan_inf_check'].get('has_issues'):
                        nan_pct = var_results['nan_inf_check']['nan_percentage']
                        inf_pct = var_results['nan_inf_check']['inf_percentage']
                        print(f"      NaN: {nan_pct:.2f}%, Inf: {inf_pct:.2f}%")
                    
                    if var_results['range_check'].get('has_issues'):
                        rc = var_results['range_check']
                        print(f"      Out of range: {rc['below_min']} below min, "
                              f"{rc['above_max']} above max")
                        print(f"        Expected: {rc['expected_range']}, "
                              f"Actual: {rc['actual_range']}")
                    
                    if var_results.get('stationarity_check', {}).get('has_issues'):
                        sc = var_results['stationarity_check']
                        if sc.get('stationarity_check'):
                            print(f"      High inter-annual variability detected")
                            print(f"        Mean CV: {sc.get('mean_cv', 'N/A'):.3f}, "
                                  f"Std CV: {sc.get('std_cv', 'N/A'):.3f} "
                                  f"(across {sc.get('n_years', 'N/A')} years)")
                    
                    if var_results.get('variance_check', {}).get('has_issues'):
                        vc = var_results['variance_check']
                        if vc.get('variance_check'):
                            print(f"      Heterogeneous variance across stations")
                            print(f"        Levene p-value: {vc.get('levene_pvalue', 'N/A'):.4f}, "
                                  f"Variance ratio: {vc.get('variance_ratio', 'N/A'):.2f}")
                    
                    if var_results.get('temporal_check', {}).get('has_issues'):
                        tc = var_results['temporal_check']
                        print(f"      Large temporal jumps: {tc['large_jumps']}")
                        print(f"        Max change: {tc.get('max_diff', 'N/A'):.2f}, "
                              f"Threshold: {tc.get('threshold', 'N/A'):.2f}")
        
        print("\n" + "="*80)
    
    def create_plots(self, output_dir: str = "plots/v2"):
        if not self.results:
            print("No results available. Run check_dataset() first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating plots in {output_dir}/...")
        
        self.plot_completeness_heatmap(output_dir)
        self.plot_nan_percentages(output_dir)
        self.plot_stationarity_results(output_dir)
        self.plot_variance_results(output_dir)
        self.plot_range_violations(output_dir)
        
        print(f"All plots saved to {output_dir}/")
    
    def plot_completeness_heatmap(self, output_dir: str):
        completeness_data = {}
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                continue
            
            for var_name, var_results in file_results['variables'].items():
                if not var_results.get('nan_inf_check'):
                    continue
                
                nan_pct = var_results['nan_inf_check']['nan_percentage']
                completeness_pct = 100 - nan_pct
                if var_name not in completeness_data:
                    completeness_data[var_name] = []

                completeness_data[var_name].append(completeness_pct)
        
        if not completeness_data:
            return
        
        avg_completeness = {var: np.mean(vals) for var, vals in completeness_data.items()}
        plt.figure(figsize=(12, 6))
        vars_sorted = sorted(avg_completeness.keys(), key=lambda x: avg_completeness[x])
        values = [avg_completeness[v] for v in vars_sorted]
        colors = ['#d73027' if v < 80 else '#fee08b' if v < 95 else '#1a9850' for v in values]
        
        plt.barh(vars_sorted, values, color=colors)
        plt.xlabel('Data Completeness (%)', fontsize=12)
        plt.ylabel('Variable', fontsize=12)
        plt.title('Data Completeness by Variable (Average Across All Files)', fontsize=14, fontweight='bold')
        plt.xlim(0, 100)
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/completeness_by_variable.png", dpi=300)
        plt.close()
        print(f"  Saved completeness_by_variable.png")
    
    def plot_nan_percentages(self, output_dir: str):
        nan_data = {}
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                continue
            
            for var_name, var_results in file_results['variables'].items():
                if not var_results.get('nan_inf_check'):
                    continue
                
                nan_pct = var_results['nan_inf_check']['nan_percentage']
                if var_name not in nan_data:
                    nan_data[var_name] = []

                nan_data[var_name].append(nan_pct)
        
        if not nan_data:
            return
        
        avg_nan = {var: np.mean(vals) for var, vals in nan_data.items()}
        plt.figure(figsize=(12, 6))
        vars_sorted = sorted(avg_nan.keys(), key=lambda x: avg_nan[x], reverse=True)
        values = [avg_nan[v] for v in vars_sorted]
        
        plt.bar(range(len(vars_sorted)), values, color='steelblue')
        plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Missing Data (%)', fontsize=12)
        plt.title('Missing Data (NaN) Percentage by Variable', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/nan_percentages.png", dpi=300)
        plt.close()
        print(f"  Saved nan_percentages.png")
    
    
    def plot_stationarity_results(self, output_dir: str):
        stationarity_data = {}
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                continue
            
            for var_name, var_results in file_results['variables'].items():
                if var_results.get('is_metadata'):
                    continue
                    
                sc = var_results.get('stationarity_check', {})
                if sc.get('stationarity_check'):
                    if var_name not in stationarity_data:
                        stationarity_data[var_name] = {'mean_cvs': [], 'std_cvs': []}
                    
                    # Collect CV values
                    if 'mean_cv' in sc:
                        stationarity_data[var_name]['mean_cvs'].append(sc['mean_cv'])
                    if 'std_cv' in sc:
                        stationarity_data[var_name]['std_cvs'].append(sc['std_cv'])
        
        if not stationarity_data:
            print("  No stationarity data to plot")
            return
        
        # Calculate average CV across files
        avg_mean_cvs = {}
        for var, data in stationarity_data.items():
            if data['mean_cvs']:
                avg_mean_cvs[var] = np.mean(data['mean_cvs'])
        
        plt.figure(figsize=(12, 6))
        vars_sorted = sorted(avg_mean_cvs.keys(), key=lambda x: avg_mean_cvs[x], reverse=True)
        values = [avg_mean_cvs[v] for v in vars_sorted]
        colors = ['#d73027' if v > 0.3 else '#fee08b' if v > 0.15 else '#1a9850' for v in values]
        
        plt.bar(range(len(vars_sorted)), values, color=colors)
        plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Coefficient of Variation (Mean)', fontsize=12)
        plt.title('Inter-Annual Consistency: CV of Yearly Means', fontsize=14, fontweight='bold')
        plt.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate variability')
        plt.axhline(y=0.30, color='red', linestyle='--', alpha=0.5, label='High variability')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stationarity_results.png", dpi=300)
        plt.close()
        print(f"  Saved stationarity_results.png")
    
    def plot_variance_results(self, output_dir: str):
        variance_data = {}
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                continue
            
            for var_name, var_results in file_results['variables'].items():
                if var_results.get('is_metadata'):
                    continue
                    
                vc = var_results.get('variance_check', {})
                if vc.get('variance_check'):
                    if var_name not in variance_data:
                        variance_data[var_name] = []
                    
                    variance_ratio = vc.get('variance_ratio', 0)
                    if variance_ratio != np.inf and variance_ratio > 0:
                        variance_data[var_name].append(variance_ratio)
        
        if not variance_data:
            print("  No variance data to plot")
            return
        
        avg_variance_ratio = {var: np.mean(ratios) for var, ratios in variance_data.items()}
        plt.figure(figsize=(12, 6))
        vars_sorted = sorted(avg_variance_ratio.keys(), key=lambda x: avg_variance_ratio[x], reverse=True)
        values = [avg_variance_ratio[v] for v in vars_sorted]
        colors = ['#d73027' if v > 10 else '#fee08b' if v > 5 else '#1a9850' for v in values]
        
        plt.bar(range(len(vars_sorted)), values, color=colors)
        plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Variance Ratio (Max/Min)', fontsize=12)
        plt.title('Variance Homogeneity Across Stations (Levene Test)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Moderate heterogeneity')
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='High heterogeneity')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/variance_homogeneity.png", dpi=300)
        plt.close()
        print(f"  Saved variance_homogeneity.png")
    
    
    def plot_range_violations(self, output_dir: str):
        violation_data = {}
        for filename, file_results in self.results.items():
            if 'error' in file_results:
                continue
            
            for var_name, var_results in file_results['variables'].items():
                if not var_results.get('range_check') or not var_results['range_check'].get('range_defined'):
                    continue
                
                rc = var_results['range_check']
                if 'below_min' in rc and 'above_max' in rc:
                    if var_name not in violation_data:
                        violation_data[var_name] = {'violations': 0, 'total': 0}
                    
                    total_violations = rc['below_min'] + rc['above_max']
                    total_values = rc.get('in_range', 0) + total_violations
                    violation_data[var_name]['violations'] += total_violations
                    violation_data[var_name]['total'] += total_values
        
        if not violation_data:
            print("  No range violation data to plot")
            return
        
        avg_violations = {
            var: (data['violations'] / data['total'] * 100) if data['total'] > 0 else 0
            for var, data in violation_data.items()
        }
        
        plt.figure(figsize=(12, 6))
        vars_sorted = sorted(avg_violations.keys(), key=lambda x: avg_violations[x], reverse=True)
        values = [avg_violations[v] for v in vars_sorted]
        
        plt.bar(range(len(vars_sorted)), values, color='#d73027')
        plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Out-of-Range Values (%)', fontsize=12)
        plt.title('Physical Range Violations by Variable', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/range_violations.png", dpi=300)
        plt.close()
        print(f"  Saved range_violations.png")
    
    def save_report(self, output_path: str = "numerical_checks_report.txt"):
        with open(output_path, 'w') as f:
            f.write("NUMERICAL CHECKS DETAILED REPORT\n")
            f.write("="*80 + "\n\n")
            
            for filename, file_results in self.results.items():
                if 'error' in file_results:
                    f.write(f"\nFILE: {filename}\n")
                    f.write(f"ERROR: {file_results['error']}\n")
                    continue
                
                f.write(f"\nFILE: {filename}\n")
                f.write(f"Dimensions: {file_results['dimensions']}\n")
                f.write("-"*80 + "\n")
                
                for var_name, var_results in file_results['variables'].items():
                    f.write(f"\nVariable: {var_name}\n")
                    f.write(f"  Status: {var_results.get('status', 'UNKNOWN')}\n")
                    f.write(f"  Shape: {var_results['shape']}\n")
                    f.write(f"  Dtype: {var_results['dtype']}\n")
                    
                    if var_results.get('numeric'):
                        f.write(f"\n  NaN/Inf Check:\n")
                        for k, v in var_results['nan_inf_check'].items():
                            f.write(f"    {k}: {v}\n")
                        
                        f.write(f"\n  Range Check:\n")
                        for k, v in var_results['range_check'].items():
                            f.write(f"    {k}: {v}\n")
                        
                        if not var_results.get('is_metadata'):
                            f.write(f"\n  Stationarity Check:\n")
                            for k, v in var_results.get('stationarity_check', {}).items():
                                f.write(f"    {k}: {v}\n")
                            
                            f.write(f"\n  Variance Homogeneity Check:\n")
                            for k, v in var_results.get('variance_check', {}).items():
                                f.write(f"    {k}: {v}\n")
                            
                            f.write(f"\n  Distribution Check:\n")
                            for k, v in var_results.get('distribution_check', {}).items():
                                f.write(f"    {k}: {v}\n")
                            
                            f.write(f"\n  Temporal Check:\n")
                            for k, v in var_results.get('temporal_check', {}).items():
                                f.write(f"    {k}: {v}\n")
                    
                    f.write("\n")
        
        print(f"\nDetailed report saved to: {output_path}")


class NumericalApplier:
    def __init__(self, dataset_path: str, output_path: str = "data/datasetv2"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.checker = NumericalChecker(dataset_path)
        self.cleaning_stats = {}
    
    def apply_nan_inf_filter(self, ds: xr.Dataset) -> xr.Dataset:
        ds_clean = ds.copy()
        
        for var_name in ds.data_vars:
            if np.issubdtype(ds[var_name].dtype, np.number):
                ds_clean[var_name] = xr.where(
                    np.isinf(ds[var_name]), 
                    np.nan, 
                    ds[var_name]
                )
        
        return ds_clean
    
    def apply_range_filter(self, ds: xr.Dataset, replace_with_nan: bool = True) -> xr.Dataset:
        ds_clean = ds.copy()
        for var_name in ds.data_vars:
            if var_name in self.checker.variable_ranges:
                min_val, max_val = self.checker.variable_ranges[var_name]
                if replace_with_nan:
                    ds_clean[var_name] = xr.where(
                        (ds[var_name] < min_val) | (ds[var_name] > max_val),
                        np.nan,
                        ds[var_name]
                    )
                else:
                    ds_clean[var_name] = ds[var_name].clip(min=min_val, max=max_val)
        
        return ds_clean
    
    
    def calculate_cleaning_stats(self, ds_original: xr.Dataset, ds_clean: xr.Dataset) -> Dict:
        stats = {}
        for var_name in ds_original.data_vars:
            if not np.issubdtype(ds_original[var_name].dtype, np.number):
                continue
            
            original_data = ds_original[var_name].values
            clean_data = ds_clean[var_name].values
            original_valid = np.sum(~np.isnan(original_data))
            clean_valid = np.sum(~np.isnan(clean_data))
            
            removed = original_valid - clean_valid
            removed_pct = (removed / original_valid * 100) if original_valid > 0 else 0
            
            stats[var_name] = {
                'original_valid': int(original_valid),
                'clean_valid': int(clean_valid),
                'removed': int(removed),
                'removed_percentage': removed_pct
            }
        
        return stats
    
    def apply_filters(self, ds: xr.Dataset, filter_inf: bool = True, filter_range: bool = True) -> Tuple[xr.Dataset, Dict]:
        ds_clean = ds.copy()
        print("  Applying filters:")
        if filter_inf:
            print("    Removing infinite values")
            ds_clean = self.apply_nan_inf_filter(ds_clean)
        
        if filter_range:
            print("    Filtering out-of-range values")
            ds_clean = self.apply_range_filter(ds_clean, replace_with_nan=True)
        
        stats = self.calculate_cleaning_stats(ds, ds_clean)
        return ds_clean, stats
    
    def process_file(self, nc_file: Path, filter_inf: bool = True, filter_range: bool = True) -> None:
        print(f"\nProcessing: {nc_file.name}")
        try:
            ds = xr.open_dataset(nc_file)
            ds_clean, stats = self.apply_filters(
                ds, 
                filter_inf=filter_inf,
                filter_range=filter_range
            )
            
            self.cleaning_stats[nc_file.name] = stats
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = self.output_path / nc_file.name
            ds_clean.to_netcdf(output_file)
            
            print(f"  Saved to: {output_file}")
            total_removed = sum(s['removed'] for s in stats.values())
            total_original = sum(s['original_valid'] for s in stats.values())
            overall_pct = (total_removed / total_original * 100) if total_original > 0 else 0
            print(f"  Removed {total_removed:,} values ({overall_pct:.2f}% of valid data)")
            
            ds.close()
            
        except Exception as e:
            print(f"Error processing {nc_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self, filter_inf: bool = True, filter_range: bool = True) -> None:
        if self.dataset_path.is_file():
            files = [self.dataset_path]
        elif self.dataset_path.is_dir():
            files = sorted(self.dataset_path.glob('*.nc'))
        else:
            raise ValueError(f"Path does not exist: {self.dataset_path}")
        
        if not files:
            raise ValueError(f"No NetCDF files found in {self.dataset_path}")
        
        print(f"Found {len(files)} NetCDF file(s) to process")
        print(f"Output directory: {self.output_path}")
        print(f"\nFilters enabled:")
        print(f"  Infinite values: {filter_inf}")
        print(f"  Range validation: {filter_range}")
        
        for nc_file in files:
            self.process_file(
                nc_file,
                filter_inf=filter_inf,
                filter_range=filter_range
            )
        
        self.save_summary_report()
    
    def save_summary_report(self, output_file: str = "cleaning_summary.txt"):
        output_path = self.output_path / output_file
        
        with open(output_path, 'w') as f:
            f.write("DATA CLEANING SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            for filename, stats in self.cleaning_stats.items():
                f.write(f"\nFile: {filename}\n")
                f.write("-"*80 + "\n")
                
                for var_name, var_stats in stats.items():
                    f.write(f"\n  Variable: {var_name}\n")
                    f.write(f"    Original valid values: {var_stats['original_valid']:,}\n")
                    f.write(f"    Clean valid values: {var_stats['clean_valid']:,}\n")
                    f.write(f"    Removed: {var_stats['removed']:,} "
                           f"({var_stats['removed_percentage']:.2f}%)\n")
                
                total_removed = sum(s['removed'] for s in stats.values())
                total_original = sum(s['original_valid'] for s in stats.values())
                overall_pct = (total_removed / total_original * 100) if total_original > 0 else 0
                
                f.write(f"\n  File Total:\n")
                f.write(f"    Removed: {total_removed:,} ({overall_pct:.2f}%)\n")
        
        print(f"\nCleaning summary saved to: {output_path}")
    
    def plot_cleaning_impact(self, output_dir: str = "plots/v2"):
        if not self.cleaning_stats:
            print("No cleaning statistics available.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        var_stats = {}
        for filename, stats in self.cleaning_stats.items():
            for var_name, var_data in stats.items():
                if var_name not in var_stats:
                    var_stats[var_name] = {'removed': 0, 'original': 0}
                    
                var_stats[var_name]['removed'] += var_data['removed']
                var_stats[var_name]['original'] += var_data['original_valid']
        
        var_percentages = {
            var: (data['removed'] / data['original'] * 100) if data['original'] > 0 else 0
            for var, data in var_stats.items()
        }
        
        vars_sorted = sorted(var_percentages.keys(), key=lambda x: var_percentages[x], reverse=True)
        values = [var_percentages[v] for v in vars_sorted]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(vars_sorted)), values, color='#d73027')
        plt.xticks(range(len(vars_sorted)), vars_sorted, rotation=45, ha='right')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Data Removed (%)', fontsize=12)
        plt.title('Impact of Data Cleaning: Percentage of Values Removed per Variable', fontsize=14, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cleaning_impact.png", dpi=300)
        plt.close()
        print(f"\n Saved cleaning impact plot to {output_dir}/cleaning_impact.png")


def main():
    print("Starting Numerical Checks and Data Cleaning")
    print("="*80)
    
    dataset_path = "data/datasetv1"
    
    print("\n1. Running numerical checks...")
    checker = NumericalChecker(dataset_path)
    
    checker.run_checks()
    checker.print_summary()
    checker.save_report()
    checker.create_plots()
    print("\nNumerical checks completed")
    
    print("\n" + "="*80)
    print("2. Applying filters and saving to datasetv2...")
    
    applier = NumericalApplier(dataset_path)
    
    applier.run(
        filter_inf=True,
        filter_range=True
    )
    applier.plot_cleaning_impact()
    print("\nData cleaning completed")
    print(f"   Cleaned data saved to: {applier.output_path}")


if __name__ == "__main__":
    main()