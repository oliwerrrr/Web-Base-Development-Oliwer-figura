# Internet Traffic Analysis Project

This project provides tools for analyzing internet traffic data from 2021 and 2023, comparing changes in network performance metrics over time, and visualizing the results.

## Features

- **Internet Traffic Analysis**: Analyze download/upload speeds, latency, and other network parameters
- **Data Visualization**: Generate charts and reports for better understanding of performance data
- **Year-to-Year Comparison**: Compare 2021 and 2023 data to identify trends and improvements
- **Interactive Visualization Viewer**: Browse through generated visualizations with a Qt-based viewer

## Key Findings

Our analysis of internet traffic data from 2021 and 2023 revealed several significant trends:

- **Dramatic Speed Improvements**: Download speeds increased by 150.77% (from 12.36 MB/s to 31.00 MB/s)
- **Upload Speed Growth**: Upload speeds improved by 232.81% (from 2.94 MB/s to 9.78 MB/s)
- **More Balanced Connections**: The ratio of download to upload speed decreased by 24.70%, indicating more symmetrical connections
- **Stronger Latency Impact**: Correlation between latency and download speed increased by 86.96%, suggesting greater sensitivity to network delays

## Project Structure

```
.
├── analysis_summary.md         # Summary of analysis findings
├── compare_2021_2023.py        # Script to compare data between years
├── comparison_2021_2023/       # Results of comparison analysis
├── data/                       # 2023 internet traffic data
├── data_2021/                  # 2021 internet traffic data
├── data_analysis.py            # Data analysis algorithms
├── data_processor.py           # Data processing utilities
├── internet_traffic_analysis.py # Internet traffic analysis module
├── internet_traffic_results_2021/ # Results from 2021 data analysis
├── internet_traffic_results_2023/ # Results from 2023 data analysis
├── run.py                      # Main script to run analyses
├── update_visualizations.py    # Script to update visualizations
└── visualization_viewer_qt.py  # Qt-based visualization viewer
```

## Usage

### Running Internet Traffic Analysis

To run the analysis on the 2023 data:

```bash
python run.py --traffic --data-dir data --output-dir internet_traffic_results_2023
```

To run the analysis on the 2021 data:

```bash
python run.py --traffic --data-dir data_2021 --output-dir internet_traffic_results_2021
```

### Generating Comparison Between Years

After analyzing both years' data, run the comparison script:

```bash
python compare_2021_2023.py
```

This will generate comparative visualizations and a report in the `comparison_2021_2023` directory.

### Using the Visualization Viewer

To browse the generated visualizations:

```bash
python run.py --viewer --dir internet_traffic_results_2023
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- PyQt5 (for visualization viewer)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Technical Details

The analysis pipeline includes several stages:

1. **Data Loading**: CSV data files are read and preprocessed
2. **Data Cleaning**: Handling missing values and outliers
3. **Statistical Analysis**: Calculating key metrics and correlations
4. **Visualization Generation**: Creating charts and plots to represent findings
5. **Report Generation**: Creating markdown reports summarizing results
6. **Comparative Analysis**: Computing year-over-year changes and trends

## Conclusion

The internet traffic analysis project provides comprehensive tools for understanding and visualizing network performance metrics. The comparison between 2021 and 2023 data shows dramatic improvements in internet speeds and changes in network behavior, particularly in the growing impact of latency on actual speeds.

These findings highlight the continuous evolution of internet infrastructure and changing usage patterns, offering valuable insights for network optimization and planning. 