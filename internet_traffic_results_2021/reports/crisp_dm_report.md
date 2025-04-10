# Internet Traffic Performance Analysis Report - CRISP-DM Methodology (2021)

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
- httpget: 45,780 rows, 15 columns
- httpgetmt: 1,199,755 rows, 15 columns
- httpgetmt6: 3,482 rows, 15 columns
- dlping: 2,530,196 rows, 10 columns
- webget: 11,396,961 rows, 28 columns

**Upload-related data:**
- httppost: 45,629 rows, 15 columns
- httppostmt: 1,194,194 rows, 15 columns
- httppostmt6: 3,469 rows, 15 columns
- ulping: 2,534,862 rows, 10 columns

**Additional network data:**
- dns: 50,000 rows, 9 columns
- ping: 50,000 rows, 10 columns
- udplatency: 50,000 rows, 10 columns
- udpjitter: 50,000 rows, 16 columns
- udpcloss: 50,000 rows, 7 columns

### Data Exploration

Key parameters in download-related data:
- Columns in httpgetmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures, location_id

Key parameters in upload-related data:
- Columns in httppostmt: unit_id, dtime, target, address, fetch_time, bytes_total, bytes_sec, bytes_sec_interval, warmup_time, warmup_bytes, sequence, threads, successes, failures, location_id

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

- R² score for the latency impact model: 0.0250
  - R² (coefficient of determination) measures what percentage of the variability in download speed is explained by the variability in latency. The value of 0.0250 means that only 2.5% of the variability in download speed is explained by latency, indicating a weak linear relationship.
  
- MAE for the latency impact model: 7,126,082.5791
  - MAE (Mean Absolute Error) measures the average absolute difference between actual and predicted speeds in bytes per second. The high MAE value suggests that the model is not precise in predicting download speed based on latency.

- The low R² value and high MAE indicate that in 2021, latency was not a strong predictor of data download speed, and other factors may have had a greater impact on performance.

## 5. Evaluation

### Results

- Average download speed: 12,362,686.32 bytes/sec
- Average upload speed: 2,937,953.23 bytes/sec
- Download to upload speed ratio: 4.21

- Correlation between latency and download speed: -0.1794
- Each additional 1ms of latency decreases download speed by 23.77 bytes/sec

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
