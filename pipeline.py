"""
Complete Data Pipeline for DWD SYNOP Dataset Processing

This pipeline orchestrates the entire workflow from raw data scraping to final dataset creation:
1. Scratch data and metadata from DWD
2. Pre-process raw data (duplicates, missing files, incorrect content, concatenation)
3. Create datasetv1 (NetCDF format with time series)
4. Create datasetv2 (Numerical quality checks and filtering)
5. Create datasetv3 (Remove NaN values)
6. Create datasetv4 (Gap filling with interpolation/extrapolation)

Usage:
    python pipeline.py --all                    # Run entire pipeline
    python pipeline.py --scratch                # Only scrape data
    python pipeline.py --preprocess             # Only pre-process
    python pipeline.py --v1                     # Only create datasetv1
    python pipeline.py --v2                     # Only create datasetv2
    python pipeline.py --v3                     # Only create datasetv3
    python pipeline.py --v4                     # Only create datasetv4
    python pipeline.py --from-scratch           # Run from data scraping onwards
    python pipeline.py --from-preprocess        # Run from pre-processing onwards
    python pipeline.py --from-v1                # Run from v1 onwards
    python pipeline.py --from-v2                # Run from v2 onwards
    python pipeline.py --from-v3                # Run from v3 onwards
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess


class DataPipeline:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.start_time = None
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "STEP": "📍"
        }.get(level, "")
        print(f"[{timestamp}] {prefix} {message}")
    
    def run_script(self, script_name: str, description: str) -> bool:
        self.log(f"Starting: {description}", "STEP")
        script_path = self.base_dir / script_name
        
        if not script_path.exists():
            self.log(f"Script not found: {script_name}", "ERROR")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout:
                print(result.stdout)
            
            self.log(f"Completed: {description}", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"Failed: {description}", "ERROR")
            if e.stdout:
                print("STDOUT:", e.stdout)

            if e.stderr:
                print("STDERR:", e.stderr)

            return False
        except Exception as e:
            self.log(f"Unexpected error in {script_name}: {str(e)}", "ERROR")
            return False
    
    def check_prerequisites(self) -> bool:
        self.log("Checking prerequisites...", "INFO")
        scripts_dir = self.base_dir / "scripts"
        if not scripts_dir.exists():
            self.log("scripts/ directory not found", "ERROR")
            return False
        
        required_scripts = [
            "scripts/scratch.py",
            "scripts/scratch_metadata.py",
            "scripts/pre-process.py",
            "scripts/create_datasetv1.py",
            "scripts/create_datasetv2.py",
            "scripts/create_datasetv3.py",
            "scripts/create_datasetv4.py"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            if not (self.base_dir / script).exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            self.log(f"Missing scripts: {', '.join(missing_scripts)}", "ERROR")
            return False
        
        self.log("All required scripts found", "SUCCESS")
        return True
    
    def step_scratch_data(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 1: Scraping Raw Data from DWD", "STEP")
        self.log("=" * 80, "INFO")
        return self.run_script("scripts/scratch.py", "Scrape raw meteorological data")
    
    def step_scratch_metadata(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 2: Scraping Metadata from DWD", "STEP")
        self.log("=" * 80, "INFO")
        return self.run_script("scripts/scratch_metadata.py", "Scrape station metadata")
    
    def step_preprocess(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 3: Pre-processing Raw Data", "STEP")
        self.log("=" * 80, "INFO")
        self.log("This step handles:", "INFO")
        self.log("  - Duplicate detection and removal", "INFO")
        self.log("  - Missing file detection and creation", "INFO")
        self.log("  - Incorrect content detection and fixing", "INFO")
        self.log("  - Station file concatenation", "INFO")
        return self.run_script("scripts/pre-process.py", "Pre-process raw data")
    
    def step_create_v1(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 4: Creating Dataset v1 (NetCDF Format)", "STEP")
        self.log("=" * 80, "INFO")
        self.log("Converts pre-processed data to NetCDF format", "INFO")
        self.log("Output: data/datasetv1/*.nc (one file per year)", "INFO")
        return self.run_script("scripts/create_datasetv1.py", "Create datasetv1")
    
    def step_create_v2(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 5: Creating Dataset v2 (Quality Checks)", "STEP")
        self.log("=" * 80, "INFO")
        self.log("Performs numerical quality checks:", "INFO")
        self.log("  - NaN and Infinity detection", "INFO")
        self.log("  - Physical range validation", "INFO")
        self.log("  - Inter-annual consistency (stationarity)", "INFO")
        self.log("  - Variance homogeneity across stations", "INFO")
        self.log("  - Distribution analysis", "INFO")
        self.log("  - Temporal consistency", "INFO")
        self.log("Output: data/datasetv2/*.nc + reports + plots", "INFO")
        return self.run_script("scripts/create_datasetv2.py", "Create datasetv2 with quality checks")
    
    def step_create_v3(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 6: Creating Dataset v3 (NaN Removal)", "STEP")
        self.log("=" * 80, "INFO")
        self.log("Removes stations/variables with excessive NaN values", "INFO")
        self.log("Output: data/datasetv3/*.nc", "INFO")
        return self.run_script("scripts/create_datasetv3.py", "Create datasetv3")
    
    def step_create_v4(self) -> bool:
        self.log("=" * 80, "INFO")
        self.log("STEP 7: Creating Dataset v4 (Gap Filling)", "STEP")
        self.log("=" * 80, "INFO")
        self.log("Fills remaining gaps using:", "INFO")
        self.log("  - Linear interpolation (gaps up to 6 values)", "INFO")
        self.log("  - Linear extrapolation (gaps up to 2 values)", "INFO")
        self.log("Output: data/datasetv4/*.nc + gap filling report", "INFO")
        return self.run_script("scripts/create_datasetv4.py", "Create datasetv4 with gap filling")
    
    def run_full_pipeline(self) -> bool:
        self.start_time = datetime.now()
        self.log("=" * 80, "INFO")
        self.log("STARTING COMPLETE DATA PIPELINE", "STEP")
        self.log("=" * 80, "INFO")
        
        steps = [
            ("Scratch Data", self.step_scratch_data),
            ("Scratch Metadata", self.step_scratch_metadata),
            ("Pre-process", self.step_preprocess),
            ("Create v1", self.step_create_v1),
            ("Create v2", self.step_create_v2),
            ("Create v3", self.step_create_v3),
            ("Create v4", self.step_create_v4),
        ]
        
        for step_name, step_func in steps:
            if not step_func():
                self.log(f"Pipeline failed at step: {step_name}", "ERROR")
                return False
        
        self.print_summary()
        return True
    
    def run_from_step(self, start_step: str) -> bool:
        self.start_time = datetime.now()
        
        step_map = {
            "scratch": [
                ("Scratch Data", self.step_scratch_data),
                ("Scratch Metadata", self.step_scratch_metadata),
                ("Pre-process", self.step_preprocess),
                ("Create v1", self.step_create_v1),
                ("Create v2", self.step_create_v2),
                ("Create v3", self.step_create_v3),
                ("Create v4", self.step_create_v4),
            ],
            "preprocess": [
                ("Pre-process", self.step_preprocess),
                ("Create v1", self.step_create_v1),
                ("Create v2", self.step_create_v2),
                ("Create v3", self.step_create_v3),
                ("Create v4", self.step_create_v4),
            ],
            "v1": [
                ("Create v1", self.step_create_v1),
                ("Create v2", self.step_create_v2),
                ("Create v3", self.step_create_v3),
                ("Create v4", self.step_create_v4),
            ],
            "v2": [
                ("Create v2", self.step_create_v2),
                ("Create v3", self.step_create_v3),
                ("Create v4", self.step_create_v4),
            ],
            "v3": [
                ("Create v3", self.step_create_v3),
                ("Create v4", self.step_create_v4),
            ],
            "v4": [
                ("Create v4", self.step_create_v4),
            ],
        }
        
        if start_step not in step_map:
            self.log(f"Unknown step: {start_step}", "ERROR")
            return False
        
        self.log(f"Running pipeline from step: {start_step}", "INFO")
        
        for step_name, step_func in step_map[start_step]:
            if not step_func():
                self.log(f"Pipeline failed at step: {step_name}", "ERROR")
                return False
        
        self.print_summary()
        return True
    
    def print_summary(self):
        if self.start_time:
            duration = datetime.now() - self.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.log("=" * 80, "INFO")
            self.log("PIPELINE COMPLETED SUCCESSFULLY", "SUCCESS")
            self.log("=" * 80, "INFO")
            self.log(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s", "INFO")
            self.log("=" * 80, "INFO")
            self.log("Output Locations:", "INFO")
            self.log("  Raw data:        data/*/unzips/", "INFO")
            self.log("  Metadata:        data/metadata/unzips/", "INFO")
            self.log("  Concatenated:    data/*/concatenated/", "INFO")
            self.log("  Dataset v1:      data/datasetv1/*.nc", "INFO")
            self.log("  Dataset v2:      data/datasetv2/*.nc", "INFO")
            self.log("  Dataset v3:      data/datasetv3/*.nc", "INFO")
            self.log("  Dataset v4:      data/datasetv4/*.nc", "INFO")
            self.log("  Reports:         *.txt", "INFO")
            self.log("  Plots:           plots/v*/", "INFO")
            self.log("=" * 80, "INFO")


def main():
    parser = argparse.ArgumentParser(
        description="DWD SYNOP Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --all          # Run complete pipeline
  python pipeline.py --scratch      # Only scrape data
  python pipeline.py --v2           # Only create datasetv2
  python pipeline.py --from-v2      # Run from v2 onwards
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--scratch", action="store_true", help="Only scrape data")
    parser.add_argument("--metadata", action="store_true", help="Only scrape metadata")
    parser.add_argument("--preprocess", action="store_true", help="Only pre-process")
    parser.add_argument("--v1", action="store_true", help="Only create datasetv1")
    parser.add_argument("--v2", action="store_true", help="Only create datasetv2")
    parser.add_argument("--v3", action="store_true", help="Only create datasetv3")
    parser.add_argument("--v4", action="store_true", help="Only create datasetv4")
    parser.add_argument("--from-scratch", action="store_true", help="Run from scratch onwards")
    parser.add_argument("--from-preprocess", action="store_true", help="Run from preprocess onwards")
    parser.add_argument("--from-v1", action="store_true", help="Run from v1 onwards")
    parser.add_argument("--from-v2", action="store_true", help="Run from v2 onwards")
    parser.add_argument("--from-v3", action="store_true", help="Run from v3 onwards")
    
    args = parser.parse_args()
    
    pipeline = DataPipeline()
    
    # Check prerequisites
    if not pipeline.check_prerequisites():
        sys.exit(1)
    
    # Determine what to run
    success = False
    
    if args.all:
        success = pipeline.run_full_pipeline()
    elif args.scratch:
        success = pipeline.step_scratch_data()
    elif args.metadata:
        success = pipeline.step_scratch_metadata()
    elif args.preprocess:
        success = pipeline.step_preprocess()
    elif args.v1:
        success = pipeline.step_create_v1()
    elif args.v2:
        success = pipeline.step_create_v2()
    elif args.v3:
        success = pipeline.step_create_v3()
    elif args.v4:
        success = pipeline.step_create_v4()
    elif args.from_scratch:
        success = pipeline.run_from_step("scratch")
    elif args.from_preprocess:
        success = pipeline.run_from_step("preprocess")
    elif args.from_v1:
        success = pipeline.run_from_step("v1")
    elif args.from_v2:
        success = pipeline.run_from_step("v2")
    elif args.from_v3:
        success = pipeline.run_from_step("v3")
    else:
        parser.print_help()
        sys.exit(0)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()