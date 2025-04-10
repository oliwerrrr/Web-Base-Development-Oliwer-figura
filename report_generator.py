#!/usr/bin/env python3
"""
Module for generating the CRISP-DM report from internet traffic analysis results.
"""

import os
import numpy as np

def generate_crisp_dm_report(results, output_dir):
    """
    Generates a report conforming to the CRISP-DM methodology using pre-computed results.
    
    Args:
        results (dict): A dictionary containing the results from InternetTrafficAnalyzer analyses. 
                        Expected keys: 'summary', 'download_performance', 'upload_performance', 
                                       'comparison', 'factor_analysis'.
        output_dir (str): The base directory where the 'reports' subfolder exists or will be created.
    
    Returns:
        str: Path to the generated report file, or None if generation failed.
    """
    reports_folder = os.path.join(output_dir, 'reports')
    if not os.path.exists(reports_folder):
        try:
            os.makedirs(reports_folder)
            print(f"Created reports directory: {reports_folder}")
        except OSError as e:
            print(f"Error creating reports directory {reports_folder}: {e}")
            return None
            
    report_path = os.path.join(reports_folder, 'crisp_dm_report.md')
    
    print(f"\nGenerating CRISP-DM report at: {report_path}...")

    # Extract results for easier access (handle potentially missing keys gracefully)
    summary = results.get('summary', {})
    factors_results = results.get('factor_analysis', {})
    comparison_results = results.get('comparison', {})
    dl_perf_results = results.get('download_performance', {})
    ul_perf_results = results.get('upload_performance', {})
    
    # Determine primary keys used (assuming they are standard, adjust if needed)
    primary_dl_key = 'httpget' 
    primary_ul_key = 'httppost'

    try:
        with open(report_path, 'w') as f:
            f.write("# Internet Traffic Performance Analysis Report (CRISP-DM Methodology)\n\n")
            
            # --- 1. Business Understanding ---
            f.write("## 1. Business Understanding\n\n")
            f.write("### Business Objectives\n\n")
            f.write("- Understand internet traffic performance (download, upload) during the analyzed period.\n")
            f.write("- Identify key network factors (e.g., latency, jitter) impacting download speed.\n")
            f.write("- Model the relationship between network parameters and performance.\n")
            f.write("- Compare characteristics of download and upload traffic.\n\n")
            
            f.write("### Success Criteria\n\n")
            f.write("- Determine Key Performance Indicators (KPIs) for internet traffic (mean/median speeds).\n")
            f.write("- Find statistically relevant correlations (> |0.1|) between network parameters (latency, jitter) and download speed.\n")
            # Check if regression was successful before stating the aim
            regression_success = 'multivariate_regression' in factors_results and factors_results['multivariate_regression']
            if regression_success:
                r2_score_val = factors_results['multivariate_regression'].get('r2_score', -1)
                f.write(f"- Create a multivariate regression model using available factors (latency, jitter) explaining download speed variability (Achieved R²: {r2_score_val:.4f}, Aim: R² > 0.1).\n")
            else:
                f.write("- Aim to create a multivariate regression model using available factors (latency, jitter) explaining download speed variability (aim for R² > 0.1) - Model not successfully generated.\n")
            f.write("- Document the process and results following CRISP-DM.\n\n")
            
            # --- 2. Data Understanding ---
            f.write("## 2. Data Understanding\n\n")
            f.write("### Data Description\n\n")
            # We don't have self.data_dir here, maybe pass it or get from summary if available?
            # For now, stating that data source info is in the analysis logs.
            f.write("Analysis uses data loaded by `InternetTrafficAnalyzer` (see execution logs for source directory).\n") 
            f.write("Main data categories loaded:\n\n")
            
            summary_dl = summary.get('download', {})
            summary_ul = summary.get('upload', {})
            summary_add = summary.get('additional', {})

            f.write("**Download Data:**\n")
            if summary_dl:
                 for key, info in summary_dl.items():
                     shape = info.get('shape', ('N/A', 'N/A'))
                     f.write(f"- `{key}`: {shape[0]} rows, {shape[1]} columns\n")
            else: f.write("- No download data summarized.\n")

            f.write("\n**Upload Data:**\n")
            if summary_ul:
                 for key, info in summary_ul.items():
                     shape = info.get('shape', ('N/A', 'N/A'))
                     f.write(f"- `{key}`: {shape[0]} rows, {shape[1]} columns\n")
            else: f.write("- No upload data summarized.\n")

            f.write("\n**Auxiliary Network Data:**\n")
            if summary_add:
                 for key, info in summary_add.items():
                     shape = info.get('shape', ('N/A', 'N/A'))
                     f.write(f"- `{key}`: {shape[0]} rows, {shape[1]} columns (full data loaded)\n")
            else: f.write("- No auxiliary data summarized.\n")

            f.write("\n### Data Exploration (Initial)\n\n")
            
            dl_key_to_show = primary_dl_key if primary_dl_key in summary_dl else list(summary_dl.keys())[0] if summary_dl else None
            ul_key_to_show = primary_ul_key if primary_ul_key in summary_ul else list(summary_ul.keys())[0] if summary_ul else None
            
            f.write("Key parameters in download data:\n")
            if dl_key_to_show and dl_key_to_show in summary_dl:
                 cols = summary_dl[dl_key_to_show].get('columns', ['N/A'])
                 f.write(f"- Key columns in `{dl_key_to_show}`: {', '.join(cols)}\n")
            else: f.write("- No primary download data to explore.\n")
            
            f.write("\nKey parameters in upload data:\n")
            if ul_key_to_show and ul_key_to_show in summary_ul:
                 cols = summary_ul[ul_key_to_show].get('columns', ['N/A'])
                 f.write(f"- Key columns in `{ul_key_to_show}`: {', '.join(cols)}\n")
            else: f.write("- No primary upload data to explore.\n")

            f.write("\nParameters in auxiliary data (examples):\n")
            ping_info = summary_add.get('ping', {})
            jitter_info = summary_add.get('udpjitter', {})
            loss_info = summary_add.get('udpcloss', {})
            if ping_info: f.write(f"- `ping`: {', '.join(ping_info.get('numeric_cols',[]))}\n")
            if jitter_info: f.write(f"- `udpjitter`: {', '.join(jitter_info.get('numeric_cols',[]))}\n")
            if loss_info: f.write(f"- `udpcloss`: {', '.join(loss_info.get('numeric_cols',[]))}\n")
            if not (ping_info or jitter_info or loss_info): f.write("- No auxiliary data examples available.\n")

            f.write("\n### Data Quality\n\n")
            f.write("- Time columns (`dtime`) and key metrics (`bytes_sec`, `rtt_avg`, `jitter_down`, etc.) were converted to appropriate types (datetime, numeric) during loading.\n")
            f.write("- Values that could not be converted were replaced with NaN (`errors='coerce'`).\n")
            f.write("- Rows with missing NaN values in key columns (`unit_id`, `dtime`, metrics) are removed before correlation and regression analysis.\n")
            f.write("- Full auxiliary data is loaded, which may require significant RAM.\n")
            
            # --- 3. Data Preparation ---
            f.write("\n## 3. Data Preparation\n\n")
            f.write("### Data Selection\n\n")
            f.write(f"- Focused on primary HTTP files (`{primary_dl_key}`, `{primary_ul_key}`) as performance indicators.\n")
            f.write(f"- Utilized `ping` (latency) and `udpjitter` (jitter) data as potential factors influencing speed.\n")
            f.write(f"- Selection between primary HTTP files (e.g., `httpget`/`httppost`) and their multi-threaded counterparts (`httpgetmt`/`httppostmt`) is done dynamically based on file size during data loading.\n\n")
            
            f.write("### Data Cleaning\n\n")
            f.write("- Outliers (below 5th and above 95th percentile) for speed (`bytes_sec`) were removed in performance analyses and comparisons.\n")
            f.write("- Rows with missing values (NaN) in `unit_id`, `dtime`, and analyzed metrics were removed before merging data and modeling.\n\n")
            
            f.write("### Data Transformation\n\n")
            f.write("- Conversion of `dtime` columns to datetime format.\n")
            f.write("- Conversion of metric columns (`bytes_sec`, `rtt_avg`, `jitter_down`) to numeric type.\n")
            f.write("- Merging speed data with factor data (`ping`, `jitter`) using `merge_asof` based on `unit_id` and the nearest preceding timestamp (`dtime`) within a 5-minute tolerance. This requires time-sorting the data.\n")
            f.write("- Daily aggregation (`resample('D')`) for visualizing speed time trends.\n\n")
            
            # --- 4. Modeling ---
            f.write("## 4. Modeling\n\n")
            f.write("### Modeling Techniques\n\n")
            f.write("- Pearson correlation analysis to assess linear relationships between download speed and factors (latency, jitter).\n")
            f.write("- Visualization of relationships using scatter plots with regression lines.\n")
            # Reference the specific factors used if available
            reg_factors = factors_results.get('multivariate_regression', {}).get('factors', [])
            if reg_factors:
                 f.write(f"- Applied Multivariate Linear Regression to model the simultaneous impact of available factors (`{', '.join(reg_factors)}`) on download speed (`bytes_sec`).\n")
            else:
                 f.write("- Attempted Multivariate Linear Regression to model the impact of factors on download speed (`bytes_sec`).\n")
            f.write("- Data split into training (70%) and testing (30%) sets for evaluating the regression model.\n\n")

            # --- 5. Evaluation --- (Added section)
            f.write("## 5. Evaluation\n\n")
            
            f.write("### Download Performance\n\n")
            if dl_perf_results:
                 for key, stats in dl_perf_results.items():
                     f.write(f"**{key}:**\n")
                     for stat_name, value in stats.items():
                         f.write(f"  - {stat_name}: {value:,.2f}\n" if isinstance(value, (int, float)) else f"  - {stat_name}: {value}\n")
                     f.write("\n")
            else: f.write("- No download performance results available.\n")

            f.write("### Upload Performance\n\n")
            if ul_perf_results:
                 for key, stats in ul_perf_results.items():
                     f.write(f"**{key}:**\n")
                     for stat_name, value in stats.items():
                         f.write(f"  - {stat_name}: {value:,.2f}\n" if isinstance(value, (int, float)) else f"  - {stat_name}: {value}\n")
                     f.write("\n")
            else: f.write("- No upload performance results available.\n")

            f.write("### Download vs. Upload Comparison\n\n")
            comp_stats = comparison_results.get('comparison', {})
            if comp_stats:
                 f.write(f"- Average Download Speed: {comp_stats.get('dl_mean', np.nan):,.2f} bytes/sec\n")
                 f.write(f"- Average Upload Speed: {comp_stats.get('ul_mean', np.nan):,.2f} bytes/sec\n")
                 f.write(f"- Median Download Speed: {comp_stats.get('dl_median', np.nan):,.2f} bytes/sec\n")
                 f.write(f"- Median Upload Speed: {comp_stats.get('ul_median', np.nan):,.2f} bytes/sec\n")
                 ratio_mean = comp_stats.get('dl_ul_ratio_mean', np.nan)
                 ratio_median = comp_stats.get('dl_ul_ratio_median', np.nan)
                 f.write(f"- Mean Speed Ratio (Download/Upload): {ratio_mean:.2f}\n" if not np.isnan(ratio_mean) else "- Mean Speed Ratio (Download/Upload): N/A\n")
                 f.write(f"- Median Speed Ratio (Download/Upload): {ratio_median:.2f}\n\n" if not np.isnan(ratio_median) else "- Median Speed Ratio (Download/Upload): N/A\n\n")
            else: f.write("- No comparison results available.\n")

            f.write("### Performance Factor Analysis\n\n")
            if 'correlations' in factors_results and factors_results['correlations']:
                f.write("**Correlations with Download Speed:**\n")
                for factor, corr in factors_results['correlations'].items():
                    f.write(f"  - {factor}: {corr:.4f}\n")
                f.write("\n")
            else: f.write("- No correlation results available.\n")

            # Correctly indented if/else block for multivariate regression
            if 'multivariate_regression' in factors_results:
                reg = factors_results['multivariate_regression']
                f.write("**Multivariate Regression Model (Predicting bytes_sec):**\n")
                f.write(f"  - Factors: {', '.join(reg.get('factors', []))}\n")
                f.write("  - Coefficients:\n")
                for factor, coef in reg.get('coefficients', {}).items():
                    f.write(f"    - {factor}: {coef:.4f}\n")
                f.write(f"  - Intercept: {reg.get('intercept', np.nan):.4f}\n")
                f.write(f"  - R² score (Test): {reg.get('r2_score', np.nan):.4f}\n")
                f.write(f"  - Mean Absolute Error (Test): {reg.get('mae', np.nan):.4f}\n")
                f.write(f"  - Mean Squared Error (Test): {reg.get('mse', np.nan):.4f}\n")
                f.write(f"  - Training Set Size: {reg.get('train_size', 'N/A')}\n")
                f.write(f"  - Test Set Size: {reg.get('test_size', 'N/A')}\n\n")
                
                f.write("  - Interpretation (simplified):\n")
                for factor, coef in reg.get('coefficients', {}).items():
                    change = "decreases" if coef < 0 else "increases"
                    f.write(f"    - A 1-unit increase in `{factor}` {change} download speed by approx. {abs(coef):.2f} bytes/sec, holding other factors constant.\n")
                f.write("\n")
            else:
                f.write("- Multivariate regression model was not successfully built or results are unavailable.\n")

            # --- 6. Deployment --- (Placeholder)
            f.write("## 6. Deployment\n\n")
            f.write("- The analysis results and generated plots are stored in the specified output directory.\n")
            f.write("- The analysis script (`internet_traffic_analysis.py`) and report generator (`report_generator.py`) can be rerun with different data directories.\n")
            f.write("- The main script (`main.py`) orchestrates the loading, analysis, and report generation.\n")
            
            print("CRISP-DM report generated successfully.")
            return report_path

    except IOError as e:
        print(f"Error writing report file {report_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during report generation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    # This part would require creating dummy results for testing
    print("Report generator script executed directly. Requires results dictionary to function.")
    # Example: 
    # dummy_results = { 'summary': {...}, ... } 
    # generate_crisp_dm_report(dummy_results, 'test_output') 