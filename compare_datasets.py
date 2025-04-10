#!/usr/bin/env python3
"""
Compares internet traffic analysis results between two datasets (2021 and 2023).
Loads cached results, calculates differences, generates comparative plots, 
and produces a detailed markdown report.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap # For wrapping long labels

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

RESULTS_DIRS = {
    2021: os.path.join(PARENT_DIR, "internet_traffic_results_2021"),
    2023: os.path.join(PARENT_DIR, "internet_traffic_results_2023")
}
COMPARISON_OUTPUT_DIR = os.path.join(PARENT_DIR, "comparison_2021_2023")
RESULTS_CACHE_FILE = "analysis_results.pkl"
REPORT_FILE = os.path.join(COMPARISON_OUTPUT_DIR, "comparison_report.md")

# Ensure comparison output directory exists
if not os.path.exists(COMPARISON_OUTPUT_DIR):
    os.makedirs(COMPARISON_OUTPUT_DIR)
# --------------------

def load_results(year):
    """Loads cached results for a given year."""
    results_dir = RESULTS_DIRS.get(year)
    if not results_dir:
        print(f"Error: Results directory configuration missing for year {year}")
        return None
        
    cache_path = os.path.join(results_dir, RESULTS_CACHE_FILE)
    if not os.path.exists(cache_path):
        print(f"Error: Cached results file not found for year {year} at {cache_path}")
        return None
        
    try:
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
            print(f"Successfully loaded results for {year}.")
            return results
    except Exception as e:
        print(f"Error loading results for year {year} from {cache_path}: {e}")
        return None

def get_nested_value(data, keys, default=np.nan):
    """Safely retrieves a nested value from a dictionary."""
    try:
        value = data
        for key in keys:
            value = value[key]
        # Ensure we don't return complex objects like DataFrames if default is scalar
        if isinstance(default, (int, float, str, bool, type(None), np.number)) and not isinstance(value, (int, float, str, bool, type(None), np.number)):
             # Don't warn if it's just an empty dict we are returning for stats
             if not (isinstance(value, dict) and not value):
                 print(f"Warning: get_nested_value expected scalar for keys {keys}, got {type(value)}. Returning default.")
             return default
        return value
    except (KeyError, TypeError, IndexError):
        return default

def format_change(val1, val2):
    """Calculates percentage change and formats it."""
    if pd.isna(val1) or pd.isna(val2) or val1 == 0:
        return "N/A"
    change = ((val2 - val1) / abs(val1)) * 100 # Use abs(val1) for robustness, esp. with correlations
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.1f}%"

def generate_comparison_plots(results_2021, results_2023):
    """Generates plots comparing the two datasets."""
    print("Generating comparison plots...")
    
    # --- Plot 1: Mean/Median Speed Comparison (Bar Chart) ---
    plot_data_agg = []
    for year, results in [(2021, results_2021), (2023, results_2023)]:
        dl_perf = get_nested_value(results, ['download_performance', 'httpget'], default={})
        ul_perf = get_nested_value(results, ['upload_performance', 'httppost'], default={})
        plot_data_agg.append({
            'Year': year,
            'Metric': 'Download Mean Speed',
            'Value': get_nested_value(dl_perf, ['mean_speed'])
        })
        plot_data_agg.append({
            'Year': year,
            'Metric': 'Download Median Speed',
            'Value': get_nested_value(dl_perf, ['median_speed'])
        })
        plot_data_agg.append({
            'Year': year,
            'Metric': 'Upload Mean Speed',
            'Value': get_nested_value(ul_perf, ['mean_speed'])
        })
        plot_data_agg.append({
            'Year': year,
            'Metric': 'Upload Median Speed',
            'Value': get_nested_value(ul_perf, ['median_speed'])
        })

    df_plot_agg = pd.DataFrame(plot_data_agg).dropna()

    try:
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_plot_agg, x='Metric', y='Value', hue='Year', palette='viridis')
        plt.title('Mean and Median Speed Comparison (2021 vs 2023)')
        plt.ylabel('Speed (bytes/sec)')
        plt.xlabel('Metric')
        plt.xticks(rotation=15, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation for large numbers
        plt.tight_layout()
        plot_path = os.path.join(COMPARISON_OUTPUT_DIR, 'speed_comparison_mean_median.png') # Renamed
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Error generating mean/median speed comparison plot: {e}")

    # --- Plot 2: Speed Ratio Comparison ---
    ratio_data = []
    for year, results in [(2021, results_2021), (2023, results_2023)]:
        comp = get_nested_value(results, ['comparison', 'comparison'], default={})
        ratio_data.append({
            'Year': year,
            'Ratio Type': 'Mean Ratio (DL/UL)',
            'Value': get_nested_value(comp, ['dl_ul_ratio_mean'])
        })
        ratio_data.append({
            'Year': year,
            'Ratio Type': 'Median Ratio (DL/UL)',
            'Value': get_nested_value(comp, ['dl_ul_ratio_median'])
        })
    df_ratio = pd.DataFrame(ratio_data).dropna()
    
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_ratio, x='Ratio Type', y='Value', hue='Year', palette='magma')
        plt.title('Download/Upload Speed Ratio Comparison (2021 vs 2023)')
        plt.ylabel('Ratio (Download Speed / Upload Speed)')
        plt.xlabel('Ratio Type')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(COMPARISON_OUTPUT_DIR, 'speed_ratio_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
         print(f"Error generating speed ratio comparison plot: {e}")

    # --- Plot 3: Latency/Jitter Correlation Comparison ---
    corr_data = []
    for year, results in [(2021, results_2021), (2023, results_2023)]:
        corrs = get_nested_value(results, ['factor_analysis', 'correlations'], default={})
        corr_data.append({
            'Year': year,
            'Factor': 'Latency (rtt_avg)',
            'Correlation': get_nested_value(corrs, ['rtt_avg'])
        })
        corr_data.append({
            'Year': year,
            'Factor': 'Jitter (jitter_down)',
            'Correlation': get_nested_value(corrs, ['jitter_down'])
        })
    df_corr = pd.DataFrame(corr_data).dropna()

    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_corr, x='Factor', y='Correlation', hue='Year', palette='coolwarm')
        plt.title('Correlation with Download Speed (2021 vs 2023)')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.xlabel('Network Factor')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.axhline(0, color='grey', linewidth=0.8)
        plt.tight_layout()
        plot_path = os.path.join(COMPARISON_OUTPUT_DIR, 'factor_correlation_comparison.png') # Renamed
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
         print(f"Error generating correlation comparison plot: {e}")

    # --- NEW PLOT 4: Speed Statistics Comparison (Mean, Median, Std, Min, Max) ---
    stats_data = []
    stat_map = {'mean_speed': 'Mean', 'median_speed': 'Median', 'std_speed': 'Std Dev', 'min_speed': 'Min', 'max_speed': 'Max'}
    for year, results in [(2021, results_2021), (2023, results_2023)]:
        for direction, key in [('Download', 'httpget'), ('Upload', 'httppost')]:
            perf = get_nested_value(results, [f'{direction.lower()}_performance', key], default={})
            for stat_key, stat_name in stat_map.items():
                stats_data.append({
                    'Year': year,
                    'Direction': direction,
                    'Statistic': stat_name,
                    'Value': get_nested_value(perf, [stat_key])
                })
    df_stats = pd.DataFrame(stats_data).dropna()

    try:
        g = sns.catplot(data=df_stats, x='Statistic', y='Value', col='Direction', hue='Year', 
                        kind='bar', palette='viridis', height=5, aspect=1.2, 
                        order=['Min', 'Median', 'Mean', 'Max', 'Std Dev']) # Control order
        g.fig.suptitle('Comparison of Speed Statistics (2021 vs 2023)', y=1.03)
        g.set_axis_labels("Statistic", "Speed (bytes/sec)")
        g.set_titles("{col_name} Speed Statistics")
        g.set_xticklabels(rotation=15, ha='right')
        # Apply scientific notation formatting to y-axis after plotting
        for ax in g.axes.flat:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
        plot_path = os.path.join(COMPARISON_OUTPUT_DIR, 'speed_statistics_comparison.png')
        plt.savefig(plot_path)
        plt.close(g.fig) # Close the figure associated with the FacetGrid
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Error generating speed statistics comparison plot: {e}")

    # --- NEW PLOT 5: Regression Coefficient Comparison ---
    coeff_data = []
    for year, results in [(2021, results_2021), (2023, results_2023)]:
        coeffs = get_nested_value(results, ['factor_analysis', 'multivariate_regression', 'coefficients'], default={})
        if coeffs:
            coeff_data.append({
                'Year': year,
                'Factor': 'Latency (rtt_avg)',
                'Coefficient': get_nested_value(coeffs, ['rtt_avg'], default=0) # Default to 0 if missing
            })
            coeff_data.append({
                'Year': year,
                'Factor': 'Jitter (jitter_down)',
                'Coefficient': get_nested_value(coeffs, ['jitter_down'], default=0)
            })
    df_coeff = pd.DataFrame(coeff_data).dropna()
    
    # Check if there is data to plot
    if not df_coeff.empty:
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_coeff, x='Factor', y='Coefficient', hue='Year', palette='coolwarm_r') # Use reversed map
            plt.title('Regression Model Coefficient Comparison (2021 vs 2023)')
            plt.ylabel('Coefficient Value (Impact on bytes/sec)')
            plt.xlabel('Network Factor')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.axhline(0, color='grey', linewidth=0.8)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,3)) # Adjust sci notation threshold
            plt.tight_layout()
            plot_path = os.path.join(COMPARISON_OUTPUT_DIR, 'regression_coefficient_comparison.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Error generating regression coefficient comparison plot: {e}")
    else:
        print("Skipping regression coefficient plot: No coefficient data found.")

def main():
    results_2021 = load_results(2021)
    results_2023 = load_results(2023)

    if results_2021 is None or results_2023 is None:
        print("Cannot perform comparison due to missing results.")
        return

    # Generate comparison plots (overwrites existing ones if present)
    generate_comparison_plots(results_2021, results_2023)

    print(f"Generating detailed comparison report: {REPORT_FILE}")
    
    # --- Extract Metrics (Simplified access using results dicts) --- 
    
    # --- Generate Report --- 
    try:
        with open(REPORT_FILE, 'w') as f:
            f.write("# Comparative Analysis: Internet Traffic Performance (2021 vs 2023)\n\n")
            f.write("This report compares key performance indicators and model results derived from internet traffic data collected in 2021 and 2023.\n\n")
            
            # --- Section 1: KPI Comparison --- 
            f.write("## 1. Key Performance Indicators (KPIs) Comparison\n\n")
            f.write("Performance metrics are based on the primary multi-threaded HTTP tests (`httpgetmt`, `httppostmt`) after filtering outliers (5th-95th percentile). Speeds are in bytes per second unless otherwise noted.\n\n")
            
            f.write("### 1.1. Summary Table\n\n")
            f.write("| Metric                   | 2021 Value        | 2023 Value        | Change (2021->2023) |\n")
            f.write("| :----------------------- | :---------------- | :---------------- | :------------------ |\n")
            
            # Extract values using get_nested_value for safety
            dl_perf_21 = get_nested_value(results_2021, ['download_performance', 'httpget'], default={})
            ul_perf_21 = get_nested_value(results_2021, ['upload_performance', 'httppost'], default={})
            comp_21 = get_nested_value(results_2021, ['comparison', 'comparison'], default={})
            dl_perf_23 = get_nested_value(results_2023, ['download_performance', 'httpget'], default={})
            ul_perf_23 = get_nested_value(results_2023, ['upload_performance', 'httppost'], default={})
            comp_23 = get_nested_value(results_2023, ['comparison', 'comparison'], default={})

            # Download Speed
            dl_mean_21 = get_nested_value(dl_perf_21, ['mean_speed'])
            dl_mean_23 = get_nested_value(dl_perf_23, ['mean_speed'])
            f.write(f"| Mean Download Speed    | {dl_mean_21:,.2f}   | {dl_mean_23:,.2f}   | {format_change(dl_mean_21, dl_mean_23)}          |\n")
            dl_med_21 = get_nested_value(dl_perf_21, ['median_speed'])
            dl_med_23 = get_nested_value(dl_perf_23, ['median_speed'])
            f.write(f"| Median Download Speed  | {dl_med_21:,.2f}   | {dl_med_23:,.2f}   | {format_change(dl_med_21, dl_med_23)}          |\n")
            dl_std_21 = get_nested_value(dl_perf_21, ['std_speed'])
            dl_std_23 = get_nested_value(dl_perf_23, ['std_speed'])
            f.write(f"| Std Dev Download Speed | {dl_std_21:,.2f}   | {dl_std_23:,.2f}   | {format_change(dl_std_21, dl_std_23)}          |\n")
            
            # Upload Speed
            ul_mean_21 = get_nested_value(ul_perf_21, ['mean_speed'])
            ul_mean_23 = get_nested_value(ul_perf_23, ['mean_speed'])
            f.write(f"| Mean Upload Speed      | {ul_mean_21:,.2f}   | {ul_mean_23:,.2f}   | {format_change(ul_mean_21, ul_mean_23)}          |\n")
            ul_med_21 = get_nested_value(ul_perf_21, ['median_speed'])
            ul_med_23 = get_nested_value(ul_perf_23, ['median_speed'])
            f.write(f"| Median Upload Speed    | {ul_med_21:,.2f}   | {ul_med_23:,.2f}   | {format_change(ul_med_21, ul_med_23)}          |\n")
            ul_std_21 = get_nested_value(ul_perf_21, ['std_speed'])
            ul_std_23 = get_nested_value(ul_perf_23, ['std_speed'])
            f.write(f"| Std Dev Upload Speed   | {ul_std_21:,.2f}   | {ul_std_23:,.2f}   | {format_change(ul_std_21, ul_std_23)}          |\n")

            # Speed Ratios
            ratio_mean_21 = get_nested_value(comp_21, ['dl_ul_ratio_mean'])
            ratio_mean_23 = get_nested_value(comp_23, ['dl_ul_ratio_mean'])
            f.write(f"| Mean Speed Ratio (DL/UL)| {ratio_mean_21:.2f}            | {ratio_mean_23:.2f}            | {format_change(ratio_mean_21, ratio_mean_23)}          |\n")
            ratio_med_21 = get_nested_value(comp_21, ['dl_ul_ratio_median'])
            ratio_med_23 = get_nested_value(comp_23, ['dl_ul_ratio_median'])
            f.write(f"| Median Speed Ratio (DL/UL)| {ratio_med_21:.2f}            | {ratio_med_23:.2f}            | {format_change(ratio_med_21, ratio_med_23)}          |\n")
            f.write("\n")

            # --- Section 1.2: Detailed Chart Explanations --- 
            f.write("### 1.2. Visualization & Interpretation\n\n")
            
            f.write("#### Mean and Median Speed Comparison\n\n")
            f.write("![Mean and Median Speed Comparison](speed_comparison_mean_median.png)\n")
            f.write("*   **What it shows:** This bar chart compares the average (mean) and typical (median) download and upload speeds between 2021 (dark purple) and 2023 (yellow). The Y-axis represents speed in bytes per second (using scientific notation, e.g., 1.0e7 = 10 million bytes/sec or ~80 Mbps).\n")
            f.write("*   **Observations:** There is a dramatic increase in all four metrics from 2021 to 2023.\n")
            f.write(f"    *   Mean download speed increased significantly (approx. {dl_mean_21/1e6:.1f} MB/s to {dl_mean_23/1e6:.1f} MB/s, a {format_change(dl_mean_21, dl_mean_23)} change).\n")
            f.write(f"    *   Median download speed also rose substantially (approx. {dl_med_21/1e6:.1f} MB/s to {dl_med_23/1e6:.1f} MB/s, a {format_change(dl_med_21, dl_med_23)} change).\n")
            f.write(f"    *   Mean upload speed saw a large increase (approx. {ul_mean_21/1e6:.1f} MB/s to {ul_mean_23/1e6:.1f} MB/s, a {format_change(ul_mean_21, ul_mean_23)} change).\n")
            f.write(f"    *   Median upload speed increased considerably (approx. {ul_med_21/1e6:.1f} MB/s to {ul_med_23/1e6:.1f} MB/s, a {format_change(ul_med_21, ul_med_23)} change).\n")
            f.write("*   **Interpretation:** Network performance, both for downloads and uploads, appears to have markedly improved between the 2021 and 2023 measurement periods based on these central tendency metrics. The improvements are substantial across the board.\n\n")
            
            f.write("#### Comparison of Speed Statistics (Including Variability)\n\n")
            f.write("![Comparison of Speed Statistics](speed_statistics_comparison.png)\n")
            f.write("*   **What it shows:** These grouped bar charts provide a more detailed view by comparing not only the mean and median, but also the minimum, maximum, and standard deviation (Std Dev) of speeds for both download (left panel) and upload (right panel) across 2021 and 2023.\n")
            f.write("*   **Observations (Download):** \n")
            f.write(f"    *   Confirms the increase in Mean and Median download speeds in 2023.\n")
            f.write(f"    *   The Maximum recorded download speed also appears significantly higher in 2023.\n")
            f.write(f"    *   The Minimum download speed might show less dramatic changes (or remain near zero).\n")
            f.write(f"    *   The Standard Deviation for download speed increased substantially in 2023 ({format_change(dl_std_21, dl_std_23)}), indicating **greater variability** in download performance in 2023 compared to 2021, despite the higher average.\n")
            f.write("*   **Observations (Upload):** \n")
            f.write(f"    *   Confirms the increase in Mean and Median upload speeds in 2023.\n")
            f.write(f"    *   Maximum upload speed is also significantly higher in 2023.\n")
            f.write(f"    *   Minimum upload speed likely shows less change.\n")
            f.write(f"    *   The Standard Deviation for upload speed also increased markedly in 2023 ({format_change(ul_std_21, ul_std_23)}), suggesting **greater variability** here as well.\n")
            f.write("*   **Interpretation:** While average and typical speeds improved significantly in 2023, the performance also became more variable (larger standard deviation). This means users might experience a wider range of speeds in 2023, with higher peaks but potentially still some slower measurements, compared to a potentially more consistent (though slower) experience in 2021.\n\n")

            f.write("#### Download/Upload Speed Ratio Comparison\n\n")
            f.write("![Download/Upload Speed Ratio Comparison](speed_ratio_comparison.png)\n")
            f.write("*   **What it shows:** This chart compares the ratio of Download Speed to Upload Speed (calculated using both mean and median values) between 2021 and 2023.\n")
            f.write("*   **Observations:** The ratio appears to have increased slightly for both mean and median comparisons from 2021 to 2023.\n")
            f.write(f"    *   Mean Ratio changed from {ratio_mean_21:.2f} to {ratio_mean_23:.2f} ({format_change(ratio_mean_21, ratio_mean_23)}).\n")
            f.write(f"    *   Median Ratio changed from {ratio_med_21:.2f} to {ratio_med_23:.2f} ({format_change(ratio_med_21, ratio_med_23)}).\n")
            f.write("*   **Interpretation:** A ratio greater than 1 indicates asymmetric performance (downloads faster than uploads). The increase in the ratio suggests that while both download and upload speeds improved, the **download speed improvement was proportionally larger**, leading to slightly *greater* asymmetry in 2023 compared to 2021. This is typical for many consumer internet connections.\n\n")

            # --- Section 2: Factor Analysis Comparison --- 
            f.write("## 2. Performance Factor Analysis Comparison\n\n")
            f.write("Comparison of how network factors (latency, jitter) correlate with download speed and the estimated impact from the multivariate regression model.\n\n")
            
            f.write("### 2.1. Summary Table\n\n")
            f.write("| Metric                     | 2021 Value | 2023 Value | Change (2021->2023) |\n")
            f.write("| :------------------------- | :--------- | :--------- | :------------------ |\n")

            # Extract factor analysis values
            factors_21 = get_nested_value(results_2021, ['factor_analysis'], default={})
            corrs_21 = get_nested_value(factors_21, ['correlations'], default={})
            reg_21 = get_nested_value(factors_21, ['multivariate_regression'], default={})
            coeffs_21 = get_nested_value(reg_21, ['coefficients'], default={})
            factors_23 = get_nested_value(results_2023, ['factor_analysis'], default={})
            corrs_23 = get_nested_value(factors_23, ['correlations'], default={})
            reg_23 = get_nested_value(factors_23, ['multivariate_regression'], default={})
            coeffs_23 = get_nested_value(reg_23, ['coefficients'], default={})

            # Correlations
            corr_lat_21 = get_nested_value(corrs_21, ['rtt_avg'])
            corr_lat_23 = get_nested_value(corrs_23, ['rtt_avg'])
            f.write(f"| Corr: DL Speed vs Latency  | {corr_lat_21:.4f}   | {corr_lat_23:.4f}   | {format_change(corr_lat_21, corr_lat_23)}          |\n")
            corr_jit_21 = get_nested_value(corrs_21, ['jitter_down'])
            corr_jit_23 = get_nested_value(corrs_23, ['jitter_down'])
            f.write(f"| Corr: DL Speed vs Jitter   | {corr_jit_21:.4f}   | {corr_jit_23:.4f}   | {format_change(corr_jit_21, corr_jit_23)}          |\n")
            
            # Regression R²
            r2_21 = get_nested_value(reg_21, ['r2_score'])
            r2_23 = get_nested_value(reg_23, ['r2_score'])
            f.write(f"| Regression R² Score        | {r2_21:.4f}   | {r2_23:.4f}   | {format_change(r2_21, r2_23)}          |\n")

            # Coefficients (Example for Latency)
            coef_lat_21 = get_nested_value(coeffs_21, ['rtt_avg'], default=np.nan)
            coef_lat_23 = get_nested_value(coeffs_23, ['rtt_avg'], default=np.nan)
            f.write(f"| Coeff: Latency (rtt_avg)   | {coef_lat_21:,.2f} | {coef_lat_23:,.2f} | {format_change(coef_lat_21, coef_lat_23)}          |\n")
            coef_jit_21 = get_nested_value(coeffs_21, ['jitter_down'], default=np.nan)
            coef_jit_23 = get_nested_value(coeffs_23, ['jitter_down'], default=np.nan)
            f.write(f"| Coeff: Jitter (jitter_down)| {coef_jit_21:,.2f} | {coef_jit_23:,.2f} | {format_change(coef_jit_21, coef_jit_23)}          |\n")
            f.write("\n")

            # --- Section 2.2: Detailed Chart Explanations --- 
            f.write("### 2.2. Visualization & Interpretation\n\n")

            f.write("#### Correlation with Download Speed\n\n")
            f.write("![Correlation with Download Speed](factor_correlation_comparison.png)\n")
            f.write("*   **What it shows:** This bar chart compares the Pearson correlation coefficient between download speed and two key network factors: Latency (`rtt_avg`) and Jitter (`jitter_down`), across 2021 and 2023.\n")
            f.write("*   **Observations:** \n")
            f.write(f"    *   Both factors show a negative correlation in both years, as expected (higher latency/jitter generally corresponds to lower speed).\n")
            f.write(f"    *   The negative correlation with Latency is slightly stronger in 2023 ({corr_lat_23:.4f}) compared to 2021 ({corr_lat_21:.4f}).\n")
            f.write(f"    *   The negative correlation with Jitter is also stronger in 2023 ({corr_jit_23:.4f}) compared to 2021 ({corr_jit_21:.4f}).\n")
            f.write("*   **Interpretation:** While still weak overall, both latency and jitter showed a slightly more pronounced negative relationship with download speed in 2023. This *could* mean that as overall speeds increased, performance became marginally more sensitive to variations in these network quality metrics, but the effect remains small based on correlation alone.\n\n")
            
            f.write("#### Regression Model Coefficient Comparison\n\n")
            f.write("![Regression Model Coefficient Comparison](regression_coefficient_comparison.png)\n")
            f.write("*   **What it shows:** This chart compares the coefficients assigned to Latency (`rtt_avg`) and Jitter (`jitter_down`) by the multivariate linear regression model trained separately for 2021 and 2023 data. The coefficient represents the estimated change in download speed (bytes/sec) for a one-unit increase in the factor, holding the other factor constant.\n")
            f.write("*   **Observations:** \n")
            f.write(f"    *   Both coefficients are negative in both years, aligning with the correlation analysis.\n")
            f.write(f"    *   The magnitude of the negative coefficient for Latency appears significantly larger in 2023 ({coef_lat_23:,.2f}) compared to 2021 ({coef_lat_21:,.2f}).\n")
            f.write(f"    *   The magnitude of the negative coefficient for Jitter also appears considerably larger in 2023 ({coef_jit_23:,.2f}) compared to 2021 ({coef_jit_21:,.2f}).\n")
            f.write("*   **Interpretation:** According to the linear model, the estimated negative impact of a one-unit increase in both latency and jitter on download speed was substantially greater in 2023 than in 2021. For example, a 1ms increase in average RTT was associated with a larger drop in predicted download speed in the 2023 model. **However**, this must be interpreted with extreme caution due to the very low R² score of the models. While the coefficients changed, the models themselves explain very little variance (1-2.5%), meaning these estimated impacts are likely unreliable predictors in isolation.\n\n")
            
            # --- Section 3: Overall Summary --- 
            f.write("## 3. Overall Summary of Observations\n\n")
            f.write("- **Major Speed Improvement:** The most prominent finding is a substantial increase in measured download and upload speeds (both mean and median) from 2021 to 2023. This suggests significant improvements in network capacity or efficiency affecting these tests.\n")
            f.write("- **Increased Variability:** Alongside higher average speeds, the variability (standard deviation) of both download and upload speeds also increased noticeably in 2023, indicating a wider range of performance experiences.\n")
            f.write("- **Increased Asymmetry:** The gap between download and upload speeds widened slightly in 2023, with download speeds increasing proportionally more than upload speeds.\n")
            f.write("- **Weak Factor Influence:** Latency (`rtt_avg`) and Jitter (`jitter_down`) consistently show weak negative correlations with download speed. While these correlations and the corresponding regression coefficients strengthened slightly in 2023, their overall explanatory power (R²) remains minimal. These factors are not the primary drivers of speed variations in these datasets based on this linear analysis.\n")
            f.write("- **Data Considerations:** The analysis excluded 2021 `traceroute` data due to parsing errors and correctly handled empty non-MT HTTP files in 2023 by using the MT versions.\n")
            f.write("- **Model Limitations:** The simple linear regression model is inadequate for accurately predicting download speed based solely on latency and jitter.\n")

        print(f"Successfully updated comparison report: {REPORT_FILE}")

    except IOError as e:
        print(f"Error writing comparison report {REPORT_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during report generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 