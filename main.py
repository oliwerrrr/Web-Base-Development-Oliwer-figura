#!/usr/bin/env python3
"""
Main script to run internet traffic analysis and generate the report for specific datasets.
"""

import os
import argparse
import pickle
import sys

# Assuming internet_traffic_analysis.py and report_generator.py are in the same directory as main.py
from internet_traffic_analysis import InternetTrafficAnalyzer
from report_generator import generate_crisp_dm_report

# --- Configuration for Specific Datasets ---
# Determine the base directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one level up to the parent directory (e.g., the 'python' workspace directory)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

DATASETS_TO_PROCESS = [
    {
        "data_subdir": "data_2021",
        "output_subdir": "internet_traffic_results_2021",
        "year": 2021
    },
    {
        "data_subdir": "data_2023",
        "output_subdir": "internet_traffic_results_2023",
        "year": 2023
    }
]

RESULTS_CACHE_FILE = "analysis_results.pkl"
# -------------------------------------------

def run_analysis_for_dataset(data_dir, output_dir, force_rerun):
    """Runs the analysis and reporting pipeline for a single dataset."""
    
    print(f"\n{'='*20} Processing Dataset: {os.path.basename(data_dir)} {'='*20}")
    
    cache_path = os.path.join(output_dir, RESULTS_CACHE_FILE)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return False # Indicate failure

    analysis_results = None

    # --- Analysis Step ---
    if not force_rerun and os.path.exists(cache_path):
        print(f"Attempting to load cached analysis results from: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                analysis_results = pickle.load(f)
            print("Successfully loaded cached results.")
        except Exception as e:
            print(f"Could not load cached results: {e}. Rerunning analysis.")
            analysis_results = None

    if analysis_results is None:
        print("\n--- Starting Analysis --- ")
        # Use absolute paths for analyzer
        analyzer = InternetTrafficAnalyzer(data_dir=data_dir, output_dir=output_dir)
        print("Loading data...")
        analyzer.load_data() # Load the data first
        
        if not analyzer.download_data and not analyzer.upload_data:
             print(f"Error: No download or upload data loaded in {data_dir}. Cannot perform analysis.")
             # Continue to the next dataset instead of exiting the whole script
             return False # Indicate failure for this dataset
        
        print("Running all analyses...")
        analysis_results = analyzer.run_all_analyses()
        
        # Cache the results
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis_results, f)
            print(f"Analysis results cached to: {cache_path}")
        except Exception as e:
            print(f"Error caching analysis results: {e}")
        print("--- Analysis Finished ---")
            

    # --- Report Generation Step ---
    if analysis_results:
        print("\n--- Starting Report Generation --- ")
        # Pass absolute output_dir to report generator
        report_file_path = generate_crisp_dm_report(analysis_results, output_dir)
        if report_file_path:
            print(f"Report generation successful: {report_file_path}")
        else:
            print("Report generation failed.")
            return False # Indicate failure
        print("--- Report Generation Finished ---")
    else:
        print("Error: No analysis results available to generate the report.")
        return False # Indicate failure
        
    print(f"{'='*20} Finished Processing: {os.path.basename(data_dir)} {'='*20}")
    return True # Indicate success

def main():
    parser = argparse.ArgumentParser(description="Run Internet Traffic Analysis and Generate Reports for specific datasets (2021, 2023).")
    # Removed --data-dir and --output-dir arguments
    parser.add_argument("--force-rerun", action="store_true",
                        help="Force rerun of the analysis even if cached results exist.")
    # Removed --skip-analysis as the logic is now per-dataset based on cache

    args = parser.parse_args()
    
    overall_success = True
    for dataset_config in DATASETS_TO_PROCESS:
        # Construct absolute paths relative to the PARENT directory
        data_dir_abs = os.path.abspath(os.path.join(PARENT_DIR, dataset_config["data_subdir"]))
        output_dir_abs = os.path.abspath(os.path.join(PARENT_DIR, dataset_config["output_subdir"]))
        
        # Check if data directory exists before running
        if not os.path.isdir(data_dir_abs):
             print(f"Error: Data directory not found: {data_dir_abs}")
             print(f"Skipping dataset for year {dataset_config['year']}.")
             overall_success = False
             continue # Skip to the next dataset
             
        success = run_analysis_for_dataset(data_dir_abs, output_dir_abs, args.force_rerun)
        if not success:
            overall_success = False # Mark overall process as failed if any dataset fails
            print(f"\n*** Processing failed for dataset: {dataset_config['data_subdir']} ***\n")

    print("\n=======================================")
    if overall_success:
        print("All dataset processing completed successfully.")
        return 0 # Exit successfully
    else:
        print("Processing completed with one or more failures.")
        return 1 # Exit with error code

if __name__ == "__main__":
    sys.exit(main()) 