# Internet Traffic Analysis Comparison: 2021 vs 2023

## Major Performance Changes

### Download Speed
- 2021: 12.36 MB/s
- 2023: 31.00 MB/s
- Change: +150.77%

### Upload Speed
- 2021: 2.94 MB/s
- 2023: 9.78 MB/s
- Change: +232.81%

### Download to Upload Speed Ratio
- 2021: 4.21
- 2023: 3.17
- Change: -24.70%

### Latency Impact on Download Speed
- 2021: correlation -0.1794
- 2023: correlation -0.3354
- Change in correlation strength: +86.96%

### Model Quality Metrics
- **R² score** (coefficient of determination):
  - 2021: 0.0250 (only 2.5% of download speed variance explained by latency)
  - 2023: 0.1142 (11.42% of download speed variance explained by latency)
  - Change: +356.80% (significant improvement in explanatory power)

- **MAE** (Mean Absolute Error):
  - 2021: 7,126,082.58 bytes/sec
  - 2023: 5,195,904.71 bytes/sec
  - Change: -27.09% (improved prediction accuracy)

## Conclusions

- **Significant improvement in download speed** - value increased by more than 50% compared to 2021
- **Significant improvement in upload speed** - value increased by more than 50% compared to 2021
- **Stronger impact of network latency** on download speed in 2023
- **Lower disparity** between download and upload speeds in 2023
- **Improved model quality** - latency became a much better predictor of download speeds in 2023, with higher explanatory power (R²) and lower prediction errors (MAE)
