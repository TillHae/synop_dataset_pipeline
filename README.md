# DWD SYNOP Dataset Processing Pipeline

Complete pipeline for processing German Weather Service (DWD) SYNOP 10-minute meteorological data into high-quality, gap-filled NetCDF datasets.

**Executing the full pipeline requires approximately 1 TB of disk space. For reliable and efficient performance, a minimum of 64 GB of RAM and 32 CPU cores is recommended.**

## Pipeline Overview

```
Raw Data (DWD) → [1] Scratch → [2] Pre-process → [3] Dataset v1 (NetCDF) → 
[4] Dataset v2 (Quality Checks) → [5] Dataset v3 (NaN Removal) → [6] Dataset v4 (Gap Filling)
```

## Quick Start

Run the complete pipeline:
```bash
python pipeline.py --all
```

Run specific stages (see `pipeline.py` main function for more details):
```bash
python pipeline.py --scratch      # Download data
python pipeline.py --v2           # Quality checks
python pipeline.py --from-v2      # Run from v2 onwards
```

## Project Structure

```
synop/
├── scripts/                 # Processing scripts
├── data/                    # Data files (raw, metadata, datasets v1-v4)
├── plots/                   # Generated analysis plots
├── pipeline.py              # Main orchestrator
├── utils.py                 # Shared utilities
└── requirements.txt         # Dependencies
```

## Pipeline Stages

1.  **Scratch Data** (`scripts/scratch.py`): Downloads raw data and metadata from DWD.
2.  **Pre-process** (`scripts/pre-process.py`): Cleans duplicates, fixes content, and concatenates station files.
3.  **Dataset v1** (`scripts/create_datasetv1.py`): Converts to NetCDF format `(station_id, time)`.
4.  **Dataset v2** (`scripts/create_datasetv2.py`): Applies quality checks (Range, Inter-Annual Consistency, Temporal) and filters invalid data.
5.  **Dataset v3** (`scripts/create_datasetv3.py`): Removes stations/variables with excessive missing data.
6.  **Dataset v4** (`scripts/create_datasetv4.py`): Fills gaps using interpolation and extrapolation.

## Requirements

```bash
pip install -r requirements.txt
```