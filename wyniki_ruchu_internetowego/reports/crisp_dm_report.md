# Internet Traffic Performance Analysis Report - CRISP-DM Methodology

## 1. Business Understanding

### Business Goals

- Understand internet traffic performance to optimize user experiences
- Identify factors affecting download and upload speeds
- Establish relationships between different network parameters

### Success Criteria

- Identification of key performance indicators for internet traffic
- Discovery of significant correlations between network parameters
- Creation of models explaining the impact of various factors on performance

## 2. Data Understanding

### Data Description

The analysis uses the following datasets related to internet traffic:

**Download-related data:**
- httpgetmt: 724,511 rows, 14 columns (source: data/curr_httpgetmt.csv)
- httpgetmt6: 489 rows, 14 columns (source: data/curr_httpgetmt6.csv)
- dlping: 1,246,832 rows, 9 columns (source: data/curr_dlping.csv)
- webget: 9,522,842 rows, 27 columns (source: data/curr_webget.csv)

**Upload-related data:**
- httppostmt: 722,118 rows, 14 columns (source: data/curr_httppostmt.csv)
- httppostmt6: 1,159 rows, 14 columns (source: data/curr_httppostmt6.csv)
- ulping: 1,260,944 rows, 9 columns (source: data/curr_ulping.csv)

**Additional network data:**
- dns: 50,000 rows, 8 columns (source: data/curr_dns.csv, partial sample)
- ping: 50,000 rows, 9 columns (source: data/curr_ping.csv, partial sample)
- traceroute: 50,000 rows, 13 columns (source: data/curr_traceroute.csv, partial sample)
- udplatency: 50,000 rows, 9 columns (source: data/curr_udplatency.csv, partial sample)
- udpjitter: 50,000 rows, 15 columns (source: data/curr_udpjitter.csv, partial sample)
- udpcloss: 50,000 rows, 6 columns (source: data/curr_udpcloss.csv, partial sample)

### Data Exploration

Key parameters in download-related data:
- Columns in httpgetmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures

Key parameters in upload-related data:
- Columns in httppostmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures

### Data Quality

- httpgetmt data has 0 missing values (0.00% of all data)
- httppostmt data has 0 missing values (0.00% of all data)

## 3. Data Preparation

### Data Selection

- Focused on HTTP GET and HTTP POST data as key performance indicators
- Included additional network parameters (ping, jitter) for correlation analysis

### Data Cleaning

- Removed outliers (below 5th and above 95th percentile) for download and upload speeds
- Removed rows with missing values for key parameters
- Aggregated data by day for time-series analysis

### Data Transformation

- Converted timestamp columns to datetime format
- Aggregated data by unit_id and day for correlation analysis

## 4. Modeling

### Modeling Techniques

- Used linear regression to determine the impact of latency on download speed
- Applied correlation analysis to identify relationships between parameters

### Model Evaluation

- R² score for the latency impact model: 0.1142
- MAE for the latency impact model: 5,195,904.7066

## 5. Evaluation

### Results

- Average download speed: 31,001,327.37 bytes/sec
- Average upload speed: 9,777,732.40 bytes/sec
- Download to upload speed ratio: 3.17

- Correlation between latency and download speed: -0.3354
- Each additional 1ms of latency decreases download speed by 27.44 bytes/sec

### Business Goals Evaluation

- Identified key performance indicators for internet traffic
- Determined the relationship between latency and download speed
- Compared download and upload performance

## 6. Deployment

### Deployment Plan

- Created a comprehensive report with visualizations
- Implemented analysis algorithms in Python modules
- Prepared results in a presentation-ready format

### Monitoring

- Analysis results can be visualized using the existing run.py application
- Regular repetition of the analysis for new data is recommended

## Summary

The internet traffic performance analysis revealed significant differences between download and upload speeds. The impact of network latency on download speed was identified. The results of the analysis can be used to optimize network configuration and improve user experiences.
