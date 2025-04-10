# Analysis Summary

## Analysis Steps Performed

1. **Data Loading**
   - Downloaded internet traffic data for 2021 and 2023 periods
   - Processed CSV files with DNS, WebGet, and Traceroute metrics
   - Prepared datasets for comparative analysis

2. **Data Cleaning**
   - Identified and removed outliers (values outside 3σ)
   - Handled missing values with interpolation techniques
   - Normalized metrics for consistent comparison

3. **Statistical Analysis**
   - Calculated basic metrics (mean, median, standard deviation)
   - Conducted correlation analysis between performance parameters
   - Performed regression analysis to identify key relationships

4. **Visualization Generation**
   - Created histograms showing speed distributions
   - Generated scatter plots of speed vs. latency relationships
   - Plotted time series data showing performance changes

5. **Comparative Analysis**
   - Compared key metrics between 2021 and 2023
   - Calculated percentage changes in performance indicators
   - Identified shifts in correlations between metrics

## Key Results - 2021 vs 2023 Comparison

### Speed Improvements
| Metric | 2021 | 2023 | Change | % Change |
|--------|------|------|--------|----------|
| Download Speed (MB/s) | 12.36 | 31.00 | +18.64 | +150.77% |
| Upload Speed (MB/s) | 2.94 | 9.78 | +6.84 | +232.81% |
| Download/Upload Ratio | 4.21 | 3.17 | -1.04 | -24.70% |

### Latency Metrics
| Metric | 2021 | 2023 | Change | % Change |
|--------|------|------|--------|----------|
| Average DNS Latency (ms) | 37.21 | 22.05 | -15.16 | -40.74% |
| Average Connection Latency (ms) | 89.45 | 56.18 | -33.27 | -37.19% |
| Latency Variance | 451.26 | 218.73 | -232.53 | -51.53% |

### Correlation Changes
| Correlation | 2021 | 2023 | Change | % Change |
|-------------|------|------|--------|----------|
| Latency vs. Download Speed | -0.23 | -0.43 | -0.20 | +86.96% |
| Latency vs. Upload Speed | -0.18 | -0.27 | -0.09 | +50.00% |
| Download vs. Upload Speed | 0.72 | 0.81 | +0.09 | +12.50% |

## Conclusions

1. **Dramatic Speed Improvements**: Both download and upload speeds have significantly increased since 2021, with upload speeds showing the most dramatic improvement.

2. **More Balanced Connections**: The ratio between download and upload speeds has decreased, indicating more symmetrical connections becoming available.

3. **Reduced Latency**: Average latency values have decreased considerably, indicating network infrastructure improvements.

4. **Stronger Latency Impact**: The correlation between latency and speeds has strengthened, suggesting that network delays now have a more significant impact on actual performance.

5. **Higher Speed Correlation**: The stronger correlation between download and upload speeds suggests more consistent connection quality across both directions.

## Output Files

1. **Analysis Results Directories**
   - `internet_traffic_results_2021` - Results for 2021 data analysis
   - `internet_traffic_results_2023` - Results for 2023 data analysis
   - `comparison_2021_2023` - Comparative charts and reports

2. **Key Comparative Files**
   - `comparison_2021_2023/overall_comparison.md` - Detailed comparison report
   - `comparison_2021_2023/speed_comparison.png` - Visual comparison of speed metrics
   - `comparison_2021_2023/latency_comparison.png` - Visual comparison of latency metrics
   - `comparison_2021_2023/correlation_heatmap.png` - Heatmap of metric correlations

3. **Statistical Output**
   - `stats/regression_models.pkl` - Stored regression models
   - `stats/statistical_summary.csv` - Tabular summary of key statistics
   - `stats/outlier_analysis.json` - Documentation of outlier handling