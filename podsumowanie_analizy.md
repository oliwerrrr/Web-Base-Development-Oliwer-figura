# Internet Traffic Analysis Summary

## Main Analysis Results

### Download and Upload Speeds

- **Average download speed**: 31,001,327.37 bytes/s (~31 MB/s)
- **Average upload speed**: 9,777,732.40 bytes/s (~9.8 MB/s)
- **Median download speed**: 14,366,030.00 bytes/s (~14.4 MB/s)
- **Median upload speed**: 2,168,848.00 bytes/s (~2.2 MB/s)
- **Average speed ratio (DL/UL)**: 3.17 (download is 3.17 times faster on average)
- **Median speed ratio (DL/UL)**: 6.62 (median download speed is 6.62 times higher)

### Correlations and Performance Factors

- **Correlation between latency and download speed**: -0.3354 (negative correlation)
- Each additional 1ms of latency decreases download speed by approximately 27.44 bytes/s
- The regression model indicates that latency explains about 11.42% of the variance in download speed (R² = 0.1142)

## Input Data Analysis

Various internet traffic datasets were analyzed:

- **Download data**: 
  - httpgetmt: 724,511 measurements (source: data/curr_httpgetmt.csv)
  - httpgetmt6: 489 measurements (source: data/curr_httpgetmt6.csv)
  - dlping: 1,246,832 measurements (source: data/curr_dlping.csv)
  - webget: 9,522,842 measurements (source: data/curr_webget.csv)

- **Upload data**:
  - httppostmt: 722,118 measurements (source: data/curr_httppostmt.csv)
  - httppostmt6: 1,159 measurements (source: data/curr_httppostmt6.csv)
  - ulping: 1,260,944 measurements (source: data/curr_ulping.csv)

- **Additional network data** (partially analyzed, 50,000 measurements each):
  - dns, ping, traceroute, udplatency, udpjitter, udpcloss (sources: corresponding data/curr_*.csv files)

## Key Findings

1. **Speed asymmetry** - There is a significant asymmetry between download and upload speeds, which is typical for most internet connections.

2. **Latency impact** - It was confirmed that higher latency negatively affects download speed.

3. **Speed distribution** - The speed distributions for both download and upload are right-skewed, meaning that most users experience speeds below the average.

4. **Time variability** - Time analysis shows that download and upload speeds can vary significantly across different time periods.

## Generated Visualizations

The analysis generated a series of visualizations that help to better understand the data:

1. Histograms of download and upload speed distributions
2. Time series charts of speed changes
3. Box plots comparing speed distributions
4. Scatter plot showing the correlation between latency and download speed

## Recommendations

1. **Asymmetry optimization** - Consider configurations offering more balanced download and upload speeds for users who frequently send large amounts of data.

2. **Latency minimization** - Take actions aimed at reducing network latency, which will translate into improved download speeds.

3. **Regular monitoring** - It is recommended to systematically conduct similar analyses to track changes in network performance over time.

4. **Deeper correlation analysis** - It is worth expanding the analysis to study the impact of other network parameters (e.g., jitter, packet loss) on data transfer rates. 