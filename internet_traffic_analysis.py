#!/usr/bin/env python3
"""
Moduł do analizy wydajności ruchu internetowego.
Implementuje funkcje do analizy pobierania i wysyłania pakietów
oraz ustalania zależności między parametrami.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class InternetTrafficAnalyzer:
    """Klasa implementująca algorytmy analizy ruchu internetowego."""
    
    def __init__(self, data_dir="data", output_dir="wyniki_ruchu_internetowego"):
        """
        Inicjalizacja analizatora ruchu internetowego.
        
        Args:
            data_dir (str): Katalog z danymi wejściowymi
            output_dir (str): Katalog wyjściowy dla wizualizacji i raportów
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.download_data = {}
        self.upload_data = {}
        self.additional_data = {}
        
        # Utworzenie katalogu na wyniki
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Foldery na różne typy wyników
        self.folders = {
            'eda': os.path.join(output_dir, 'eda'),
            'correlation': os.path.join(output_dir, 'correlation'),
            'regression': os.path.join(output_dir, 'regression'),
            'performance': os.path.join(output_dir, 'performance'),
            'reports': os.path.join(output_dir, 'reports')
        }
        
        # Utworzenie podkatalogów
        for folder in self.folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _get_file_size(self, file_path):
        """Helper function to get file size, returns 0 if file doesn't exist."""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def _load_csv_safe(self, file_path, **kwargs):
        """Safely loads a CSV file, handling potential errors."""
        try:
            # Using low_memory=False to prevent dtype warnings with large files
            data = pd.read_csv(file_path, low_memory=False, **kwargs)
            if not data.empty:
                print(f"Successfully loaded: {os.path.basename(file_path)} ({data.shape[0]} rows)")
                return data
            else:
                print(f"Loaded empty file: {os.path.basename(file_path)}")
                return None
        except FileNotFoundError:
            print(f"File not found: {os.path.basename(file_path)}")
            return None
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
            return None
            
    def load_data(self):
        """
        Wczytanie danych dotyczących ruchu internetowego.
        Dynamically selects main HTTP files based on size and loads full auxiliary data.
        
        Returns:
            dict: Słownik z wczytanymi danymi
        """
        print(f"Loading internet traffic data from: {self.data_dir}")

        # --- Dynamic HTTP Download File Selection ---
        httpget_path = os.path.join(self.data_dir, 'curr_httpget.csv')
        httpgetmt_path = os.path.join(self.data_dir, 'curr_httpgetmt.csv')
        httpget_size = self._get_file_size(httpget_path)
        httpgetmt_size = self._get_file_size(httpgetmt_path)
        
        # Prefer MT if significantly larger or non-MT is tiny
        if httpgetmt_size > httpget_size * 1.1 or httpget_size < 1024: 
            download_files = {
                'httpget': 'curr_httpgetmt.csv', # Primary source
                'httpget_alt': 'curr_httpget.csv', # Secondary if needed
                'httpgetmt6': 'curr_httpgetmt6.csv',
                'dlping': 'curr_dlping.csv',
                'webget': 'curr_webget.csv'
            }
            print("Prioritizing 'curr_httpgetmt.csv' for downloads based on size.")
        else:
             download_files = {
                'httpget': 'curr_httpget.csv', # Primary source
                'httpget_alt': 'curr_httpgetmt.csv', # Secondary if needed
                'httpgetmt6': 'curr_httpgetmt6.csv',
                'dlping': 'curr_dlping.csv',
                'webget': 'curr_webget.csv'
            }
             print("Prioritizing 'curr_httpget.csv' for downloads based on size.")
        
        # --- Dynamic HTTP Upload File Selection ---
        httppost_path = os.path.join(self.data_dir, 'curr_httppost.csv')
        httppostmt_path = os.path.join(self.data_dir, 'curr_httppostmt.csv')
        httppost_size = self._get_file_size(httppost_path)
        httppostmt_size = self._get_file_size(httppostmt_path)

        # Prefer MT if significantly larger or non-MT is tiny
        if httppostmt_size > httppost_size * 1.1 or httppost_size < 1024:
             upload_files = {
                'httppost': 'curr_httppostmt.csv', # Primary source
                'httppost_alt': 'curr_httppost.csv', # Secondary if needed
                'httppostmt6': 'curr_httppostmt6.csv',
                'ulping': 'curr_ulping.csv'
            }
             print("Prioritizing 'curr_httppostmt.csv' for uploads based on size.")
        else:
             upload_files = {
                'httppost': 'curr_httppost.csv', # Primary source
                'httppost_alt': 'curr_httppostmt.csv', # Secondary if needed
                'httppostmt6': 'curr_httppostmt6.csv',
                'ulping': 'curr_ulping.csv'
            }
             print("Prioritizing 'curr_httppost.csv' for uploads based on size.")

        # Wczytaj dane pobierania
        self.download_data = {}
        for key, filename in download_files.items():
            file_path = os.path.join(self.data_dir, filename)
            data = self._load_csv_safe(file_path)
            if data is not None:
                self.download_data[key] = data

        # Wczytaj dane wysyłania
        self.upload_data = {}
        for key, filename in upload_files.items():
            file_path = os.path.join(self.data_dir, filename)
            data = self._load_csv_safe(file_path)
            if data is not None:
                self.upload_data[key] = data
        
        # Wczytaj dodatkowe pliki (pełne wersje - UWAGA NA PAMIĘĆ!)
        additional_files = {
            'dns': 'curr_dns.csv',
            'ping': 'curr_ping.csv',
            'traceroute': 'curr_traceroute.csv',
            'udplatency': 'curr_udplatency.csv',
            'udpjitter': 'curr_udpjitter.csv', # Added
            'udpcloss': 'curr_udpcloss.csv'   # Added
        }
        
        self.additional_data = {}
        print("\nLoading auxiliary data (full files - may take time/memory)...")
        for key, filename in additional_files.items():
            file_path = os.path.join(self.data_dir, filename)
            # Removed nrows limit - load full file
            # Consider sampling or chunking if memory issues arise:
            # e.g., data = pd.read_csv(file_path, low_memory=False, chunksize=100000) # for chunking
            # e.g., data = pd.read_csv(file_path, low_memory=False, skiprows=lambda i: i>0 and np.random.rand() > 0.1) # for sampling 10%
            data = self._load_csv_safe(file_path) 
            if data is not None:
                self.additional_data[key] = data
                
        # Konwertuj kluczowe kolumny na numeryczne / datetime
        self._convert_data_types()
        
        print("\nData loading complete.")
        return {
            'download': self.download_data,
            'upload': self.upload_data,
            'additional': self.additional_data
        }

    def _convert_data_types(self):
        """Converts relevant columns to numeric or datetime formats."""
        print("Converting data types...")
        
        # Speed columns
        for key, data in self.download_data.items():
            if 'bytes_sec' in data.columns:
                data['bytes_sec'] = pd.to_numeric(data['bytes_sec'], errors='coerce')
        for key, data in self.upload_data.items():
            if 'bytes_sec' in data.columns:
                 data['bytes_sec'] = pd.to_numeric(data['bytes_sec'], errors='coerce')
                
        # Time columns
        for key, data in self.download_data.items():
             if 'dtime' in data.columns:
                 data['dtime'] = pd.to_datetime(data['dtime'], errors='coerce')
        for key, data in self.upload_data.items():
            if 'dtime' in data.columns:
                 data['dtime'] = pd.to_datetime(data['dtime'], errors='coerce')
        for key, data in self.additional_data.items():
             if 'dtime' in data.columns:
                 data['dtime'] = pd.to_datetime(data['dtime'], errors='coerce')

        # Additional numeric columns (assuming column names)
        if 'ping' in self.additional_data and 'rtt_avg' in self.additional_data['ping'].columns:
            self.additional_data['ping']['rtt_avg'] = pd.to_numeric(self.additional_data['ping']['rtt_avg'], errors='coerce')
        if 'udpjitter' in self.additional_data and 'avg_jitter' in self.additional_data['udpjitter'].columns:
            # Assuming 'avg_jitter' - VERIFY ACTUAL COLUMN NAME
            self.additional_data['udpjitter']['avg_jitter'] = pd.to_numeric(self.additional_data['udpjitter']['avg_jitter'], errors='coerce')
        if 'udpcloss' in self.additional_data and 'loss_ratio' in self.additional_data['udpcloss'].columns:
            # Assuming 'loss_ratio' - VERIFY ACTUAL COLUMN NAME
            self.additional_data['udpcloss']['loss_ratio'] = pd.to_numeric(self.additional_data['udpcloss']['loss_ratio'], errors='coerce')
            
        print("Data type conversion finished.")

    def generate_data_summary(self):
        """
        Generates a basic data summary.
        
        Returns:
            dict: Dictionary containing the data summary
        """
        summary = {
            'download': {},
            'upload': {},
            'additional': {} # Add summary for additional data
        }
        
        print("\nGenerating data summary...")
        # Podsumowanie danych pobierania
        for key, data in self.download_data.items():
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary['download'][key] = {
                    'shape': data.shape,
                    'columns': data.columns.tolist(),
                    'numeric_cols': numeric_cols,
                    'stats': data[numeric_cols].describe()
                }
        
        # Podsumowanie danych wysyłania
        for key, data in self.upload_data.items():
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary['upload'][key] = {
                    'shape': data.shape,
                    'columns': data.columns.tolist(),
                    'numeric_cols': numeric_cols,
                    'stats': data[numeric_cols].describe()
                }

        # Podsumowanie danych dodatkowych
        for key, data in self.additional_data.items():
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary['additional'][key] = {
                    'shape': data.shape,
                    'columns': data.columns.tolist(),
                    'numeric_cols': numeric_cols,
                    'stats': data[numeric_cols].describe()
                }
        
        # Zapisz podsumowanie do pliku
        summary_file = os.path.join(self.folders['reports'], 'data_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("# Internet Traffic Data Summary\n\n")
            
            def write_summary(title, data_dict):
                f.write(f"## {title}\n\n")
                if not data_dict:
                    f.write("No data available for summary.\n\n")
                    return
                for key, info in data_dict.items():
                    f.write(f"### {key}\n")
                    f.write(f"Shape: {info['shape']}\n")
                    f.write(f"Columns: {', '.join(info['columns'])}\n")
                    f.write(f"Numeric Columns: {', '.join(info['numeric_cols'])}\n\n")
                    f.write("Statistics:\n")
                    try:
                         f.write(f"{info['stats'].to_string()}\n\n")
                    except Exception as e:
                         f.write(f"Could not generate statistics: {e}\n\n")

            write_summary("Download Data", summary['download'])
            write_summary("Upload Data", summary['upload'])
            write_summary("Auxiliary Data", summary['additional'])

        print(f"Data summary saved to: {summary_file}")
        return summary

    def analyze_download_performance(self):
        """
        Analyzes download performance.
        Focuses on download speed and response times.
        Uses the primary download key ('httpget') determined during loading.
        
        Returns:
            dict: Results of the download performance analysis
        """
        results = {}
        primary_dl_key = 'httpget' # Key determined in load_data
        
        print(f"\nAnalyzing download performance (using primary key: {primary_dl_key})...")
        
        if primary_dl_key in self.download_data:
            data = self.download_data[primary_dl_key]
            
            if 'bytes_sec' in data.columns and pd.api.types.is_numeric_dtype(data['bytes_sec']):
                bytes_sec = data['bytes_sec'].dropna()
                
                if not bytes_sec.empty:
                    # Usuwam wartości poniżej 5 percentyla i powyżej 95 percentyla
                    lower_bound = bytes_sec.quantile(0.05)
                    upper_bound = bytes_sec.quantile(0.95)
                    print(f"Filtering {primary_dl_key} speeds between {lower_bound:.2f} (5th pct) and {upper_bound:.2f} (95th pct)")
                    
                    filtered_bytes_sec = bytes_sec[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)]
                    
                    if not filtered_bytes_sec.empty:
                        # Histogram prędkości pobierania
                        plt.figure(figsize=(10, 6))
                        sns.histplot(filtered_bytes_sec, kde=True)
                        plt.title(f'Download Speed Distribution ({primary_dl_key})')
                        plt.xlabel('Speed (bytes/sec)')
                        plt.ylabel('Number of Measurements')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], f'{primary_dl_key}_speed_distribution.png'))
                        plt.close()
                        
                        # Statystyki
                        results[primary_dl_key] = {
                            'mean_speed': filtered_bytes_sec.mean(),
                            'median_speed': filtered_bytes_sec.median(),
                            'std_speed': filtered_bytes_sec.std(),
                            'min_speed': filtered_bytes_sec.min(),
                            'max_speed': filtered_bytes_sec.max()
                        }
                        
                        # Analiza czasowa (jeśli dostępna kolumna dtime)
                        if 'dtime' in data.columns and pd.api.types.is_datetime64_any_dtype(data['dtime']):
                            try:
                                # Use data corresponding to filtered speeds for time analysis
                                filtered_data = data.loc[filtered_bytes_sec.index].copy()
                                filtered_data = filtered_data.dropna(subset=['dtime']) # Drop rows where time is invalid
                                
                                if not filtered_data.empty:
                                    # Sort before setting index for resampling
                                    filtered_data = filtered_data.sort_values('dtime') 
                                    filtered_data.set_index('dtime', inplace=True)
                                    
                                    # Agregacja dzienna
                                    daily_data = filtered_data[['bytes_sec']].resample('D').mean()
                                    
                                    if not daily_data.empty:
                                        # Wykres zmian prędkości w czasie
                                        plt.figure(figsize=(12, 6))
                                        plt.plot(daily_data.index, daily_data['bytes_sec'])
                                        plt.title(f'Average Daily Download Speed Over Time ({primary_dl_key})')
                                        plt.xlabel('Date')
                                        plt.ylabel('Speed (bytes/sec)')
                                        plt.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(self.folders['performance'], f'{primary_dl_key}_speed_timeline.png'))
                                        plt.close()
                            except Exception as e:
                                print(f"Błąd podczas analizy czasowej dla {primary_dl_key}: {e}")
                    else:
                        print(f"No data left for {primary_dl_key} after filtering outliers.")
                else:
                     print(f"No valid 'bytes_sec' data found for {primary_dl_key}.")
            else:
                 print(f"'bytes_sec' column missing or not numeric in {primary_dl_key} data.")
        else:
            print(f"Primary download data key '{primary_dl_key}' not found in loaded data.")

        # --- Analysis for 'webget' (if present) ---
        # (Keep this section similar to original, but ensure safety checks)
        if 'webget' in self.download_data:
            data = self.download_data['webget']
            if 'bytes_sec' in data.columns and pd.api.types.is_numeric_dtype(data['bytes_sec']):
                bytes_sec = data['bytes_sec'].dropna()
                if not bytes_sec.empty:
                     lower_bound = bytes_sec.quantile(0.05)
                     upper_bound = bytes_sec.quantile(0.95)
                     filtered_bytes_sec = bytes_sec[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)]
                     if not filtered_bytes_sec.empty:
                        plt.figure(figsize=(10, 6))
                        sns.histplot(filtered_bytes_sec, kde=True)
                        plt.title(f'Download Speed Distribution (Web GET)')
                        plt.xlabel('Speed (bytes/sec)')
                        plt.ylabel('Number of Measurements')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], 'webget_speed_distribution.png'))
                        plt.close()
                        results['webget'] = {
                            'mean_speed': filtered_bytes_sec.mean(),
                            'median_speed': filtered_bytes_sec.median(),
                            'std_speed': filtered_bytes_sec.std(),
                            'min_speed': filtered_bytes_sec.min(),
                            'max_speed': filtered_bytes_sec.max()
                        }
                     else: print("No 'webget' data after outlier filtering.")
                else: print("No valid 'bytes_sec' in 'webget'.")
            else: print("'bytes_sec' missing/not numeric in 'webget'.")

        # Zapisz wyniki do pliku
        report_file = os.path.join(self.folders['reports'], 'download_performance_analysis.txt')
        with open(report_file, 'w') as f:
            f.write("# Download Performance Analysis\n\n")
            if not results:
                f.write("No download performance results generated.\n")
            else:
                for key, stats in results.items():
                    f.write(f"## {key}\n\n")
                    for stat_name, value in stats.items():
                        # Format only if numeric
                        if isinstance(value, (int, float)):
                            f.write(f"{stat_name}: {value:,.2f}\n")
                        else:
                             f.write(f"{stat_name}: {value}\n")
                    f.write("\n")
        print(f"Download performance analysis saved to: {report_file}")
        return results

    def analyze_upload_performance(self):
        """
        Analyzes upload performance.
        Uses the primary upload key ('httppost') determined during loading.
        
        Returns:
            dict: Results of the upload performance analysis
        """
        results = {}
        primary_ul_key = 'httppost' # Key determined in load_data
        
        print(f"\nAnalyzing upload performance (using primary key: {primary_ul_key})...")

        if primary_ul_key in self.upload_data:
            data = self.upload_data[primary_ul_key]
            
            if 'bytes_sec' in data.columns and pd.api.types.is_numeric_dtype(data['bytes_sec']):
                bytes_sec = data['bytes_sec'].dropna()
                
                if not bytes_sec.empty:
                    lower_bound = bytes_sec.quantile(0.05)
                    upper_bound = bytes_sec.quantile(0.95)
                    print(f"Filtering {primary_ul_key} speeds between {lower_bound:.2f} (5th pct) and {upper_bound:.2f} (95th pct)")

                    filtered_bytes_sec = bytes_sec[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)]
                    
                    if not filtered_bytes_sec.empty:
                        # Histogram prędkości wysyłania
                        plt.figure(figsize=(10, 6))
                        sns.histplot(filtered_bytes_sec, kde=True)
                        plt.title(f'Upload Speed Distribution ({primary_ul_key})')
                        plt.xlabel('Speed (bytes/sec)')
                        plt.ylabel('Number of Measurements')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], f'{primary_ul_key}_speed_distribution.png'))
                        plt.close()
                        
                        # Statystyki
                        results[primary_ul_key] = {
                            'mean_speed': filtered_bytes_sec.mean(),
                            'median_speed': filtered_bytes_sec.median(),
                            'std_speed': filtered_bytes_sec.std(),
                            'min_speed': filtered_bytes_sec.min(),
                            'max_speed': filtered_bytes_sec.max()
                        }
                        
                        # Analiza czasowa
                        if 'dtime' in data.columns and pd.api.types.is_datetime64_any_dtype(data['dtime']):
                             try:
                                filtered_data = data.loc[filtered_bytes_sec.index].copy()
                                filtered_data = filtered_data.dropna(subset=['dtime'])
                                if not filtered_data.empty:
                                    filtered_data = filtered_data.sort_values('dtime')
                                    filtered_data.set_index('dtime', inplace=True)
                                    daily_data = filtered_data[['bytes_sec']].resample('D').mean()
                                    if not daily_data.empty:
                                        plt.figure(figsize=(12, 6))
                                        plt.plot(daily_data.index, daily_data['bytes_sec'])
                                        plt.title(f'Average Daily Upload Speed Over Time ({primary_ul_key})')
                                        plt.xlabel('Date')
                                        plt.ylabel('Speed (bytes/sec)')
                                        plt.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(self.folders['performance'], f'{primary_ul_key}_speed_timeline.png'))
                                        plt.close()
                             except Exception as e:
                                 print(f"Błąd podczas analizy czasowej dla {primary_ul_key}: {e}")
                    else:
                         print(f"No data left for {primary_ul_key} after filtering outliers.")
                else:
                     print(f"No valid 'bytes_sec' data found for {primary_ul_key}.")
            else:
                 print(f"'bytes_sec' column missing or not numeric in {primary_ul_key} data.")
        else:
            print(f"Primary upload data key '{primary_ul_key}' not found in loaded data.")
        
        # Zapisz wyniki do pliku
        report_file = os.path.join(self.folders['reports'], 'upload_performance_analysis.txt')
        with open(report_file, 'w') as f:
            f.write("# Upload Performance Analysis\n\n")
            if not results:
                 f.write("No upload performance results generated.\n")
            else:
                for key, stats in results.items():
                    f.write(f"## {key}\n\n")
                    for stat_name, value in stats.items():
                        if isinstance(value, (int, float)):
                             f.write(f"{stat_name}: {value:,.2f}\n")
                        else:
                             f.write(f"{stat_name}: {value}\n")
                    f.write("\n")
        print(f"Upload performance analysis saved to: {report_file}")
        return results

    def analyze_download_upload_comparison(self):
        """
        Compares download and upload performance using primary keys.
        
        Returns:
            dict: Performance comparison results
        """
        results = {'comparison': {}}
        primary_dl_key = 'httpget'
        primary_ul_key = 'httppost'
        
        print("\nComparing download vs upload performance...")

        if primary_dl_key in self.download_data and primary_ul_key in self.upload_data:
            dl_data = self.download_data[primary_dl_key]
            ul_data = self.upload_data[primary_ul_key]
            
            dl_valid = 'bytes_sec' in dl_data.columns and pd.api.types.is_numeric_dtype(dl_data['bytes_sec'])
            ul_valid = 'bytes_sec' in ul_data.columns and pd.api.types.is_numeric_dtype(ul_data['bytes_sec'])

            if dl_valid and ul_valid:
                try:
                    dl_speeds = dl_data['bytes_sec'].dropna()
                    ul_speeds = ul_data['bytes_sec'].dropna()
                    
                    if not dl_speeds.empty and not ul_speeds.empty:
                        # Usuwam wartości odstające
                        dl_lower, dl_upper = dl_speeds.quantile(0.05), dl_speeds.quantile(0.95)
                        ul_lower, ul_upper = ul_speeds.quantile(0.05), ul_speeds.quantile(0.95)
                        
                        dl_filtered = dl_speeds[(dl_speeds >= dl_lower) & (dl_speeds <= dl_upper)]
                        ul_filtered = ul_speeds[(ul_speeds >= ul_lower) & (ul_speeds <= ul_upper)]

                        if not dl_filtered.empty and not ul_filtered.empty:
                            # Statystyki
                            dl_mean = dl_filtered.mean()
                            ul_mean = ul_filtered.mean()
                            dl_median = dl_filtered.median()
                            ul_median = ul_filtered.median()
                            
                            ratio_mean = dl_mean / ul_mean if ul_mean > 0 else np.nan
                            ratio_median = dl_median / ul_median if ul_median > 0 else np.nan
                            
                            results['comparison'] = {
                                'dl_mean': dl_mean,
                                'ul_mean': ul_mean,
                                'dl_median': dl_median,
                                'ul_median': ul_median,
                                'dl_ul_ratio_mean': ratio_mean,
                                'dl_ul_ratio_median': ratio_median
                            }
                            
                            # Porównanie rozkładów - oddzielne histogramy
                            plt.figure(figsize=(12, 8))
                            
                            plt.subplot(2, 1, 1)
                            sns.histplot(dl_filtered, color='blue', kde=True, label='Download')
                            plt.title('Download Speed Distribution')
                            plt.xlabel('Speed (bytes/sec)')
                            plt.ylabel('Number of Measurements')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            plt.subplot(2, 1, 2)
                            sns.histplot(ul_filtered, color='red', kde=True, label='Upload')
                            plt.title('Upload Speed Distribution')
                            plt.xlabel('Speed (bytes/sec)')
                            plt.ylabel('Number of Measurements')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(self.folders['performance'], 'download_upload_distributions.png'))
                            plt.close()
                            
                            # Wykres pudełkowy - ograniczamy dane, aby były tej samej długości
                            sample_size = min(len(dl_filtered), len(ul_filtered), 5000) # Increased sample size
                            
                            if sample_size > 0:
                                dl_sample = dl_filtered.sample(sample_size, random_state=42)
                                ul_sample = ul_filtered.sample(sample_size, random_state=42)
                                
                                plt.figure(figsize=(8, 6))
                                box_data = pd.DataFrame({'Download': dl_sample.values, 'Upload': ul_sample.values})
                                sns.boxplot(data=box_data)
                                plt.title('Download vs Upload Speed Distribution (Sample)')
                                plt.ylabel('Speed (bytes/sec)')
                                plt.grid(True, alpha=0.3)
                                plt.savefig(os.path.join(self.folders['performance'], 'download_upload_boxplot.png'))
                                plt.close()
                            
                            # Zapisz wyniki do pliku
                            report_file = os.path.join(self.folders['reports'], 'download_upload_comparison.txt')
                            with open(report_file, 'w') as f:
                                f.write("# Download vs Upload Performance Comparison\n\n")
                                f.write("## Statistics\n\n")
                                f.write(f"Average Download Speed: {dl_mean:,.2f} bytes/sec\n")
                                f.write(f"Average Upload Speed: {ul_mean:,.2f} bytes/sec\n")
                                f.write(f"Median Download Speed: {dl_median:,.2f} bytes/sec\n")
                                f.write(f"Median Upload Speed: {ul_median:,.2f} bytes/sec\n")
                                f.write(f"Mean Speed Ratio (Download/Upload): {ratio_mean:.2f}\n" if not np.isnan(ratio_mean) else "Mean Speed Ratio (Download/Upload): N/A\n")
                                f.write(f"Median Speed Ratio (Download/Upload): {ratio_median:.2f}\n" if not np.isnan(ratio_median) else "Median Speed Ratio (Download/Upload): N/A\n")
                            print(f"Download/Upload comparison report saved to: {report_file}")
                        else:
                             print("No data remains after filtering speeds for comparison.")
                    else:
                         print("Not enough valid speed data for comparison.")
                except Exception as e:
                    print(f"Error during download/upload comparison: {e}")
            else:
                print("Missing or non-numeric 'bytes_sec' in primary download or upload data.")
        else:
             print("Primary download or upload data not available for comparison.")
        
        return results

    def analyze_performance_factors(self):
        """
        Analyzes factors affecting performance (download speed).
        Uses merge_asof for data joining and multivariate regression.
        
        Returns:
            dict: Results of the performance factor analysis
        """
        results = {}
        primary_dl_key = 'httpget'
        latency_key = 'ping'
        jitter_key = 'udpjitter'
        # loss_key = 'udpcloss'

        # Verify required columns exist (adjust names if needed)
        latency_col = 'rtt_avg'
        jitter_col = 'jitter_down' # Updated based on header inspection
        # loss_col = 'loss_ratio'  # ASSUMED NAME
        
        required_columns = {
            primary_dl_key: ['unit_id', 'dtime', 'bytes_sec'],
            latency_key: ['unit_id', 'dtime', latency_col],
            jitter_key: ['unit_id', 'dtime', jitter_col],
            # loss_key: ['unit_id', 'dtime', loss_col] # Removed loss factor for now
        }
        
        # Check if primary download data and at least one factor data is available
        if primary_dl_key not in self.download_data:
             print(f"Primary download data '{primary_dl_key}' not available for factor analysis.")
             return results
             
        dl_data = self.download_data[primary_dl_key]

        available_factors = {}
        if latency_key in self.additional_data: available_factors[latency_key] = self.additional_data[latency_key]
        if jitter_key in self.additional_data: available_factors[jitter_key] = self.additional_data[jitter_key]
        # if loss_key in self.additional_data: available_factors[loss_key] = self.additional_data[loss_key] # Removed loss factor
        
        if not available_factors:
             print("No auxiliary factor data (ping, jitter) available for analysis.")
             return results

        print("\nAnalyzing performance factors (latency, jitter)...")
        
        try:
            # --- Prepare Download Data ---
            dl_data = dl_data[required_columns[primary_dl_key]].copy()
            dl_data = dl_data.dropna()
            # Ensure types are correct
            dl_data['bytes_sec'] = pd.to_numeric(dl_data['bytes_sec'], errors='coerce')
            dl_data['dtime'] = pd.to_datetime(dl_data['dtime'], errors='coerce')
            dl_data = dl_data.dropna(subset=['dtime', 'bytes_sec', 'unit_id'])
            # Sort for merge_asof
            dl_data = dl_data.sort_values('dtime')
            
            if dl_data.empty:
                print(f"No valid download data ({primary_dl_key}) after cleaning for factor analysis.")
                return results

            # --- Prepare and Merge Factor Data ---
            merged_data = dl_data
            factor_cols = []

            for key, factor_data in available_factors.items():
                cols_to_check = required_columns[key]
                factor_metric_col = cols_to_check[-1] # e.g., rtt_avg, avg_jitter

                if not all(col in factor_data.columns for col in cols_to_check):
                    print(f"Skipping factor '{key}': Missing required columns ({', '.join(cols_to_check)}).")
                    continue
                    
                factor_data = factor_data[cols_to_check].copy()
                factor_data[factor_metric_col] = pd.to_numeric(factor_data[factor_metric_col], errors='coerce')
                factor_data['dtime'] = pd.to_datetime(factor_data['dtime'], errors='coerce')
                factor_data = factor_data.dropna()
                
                if factor_data.empty:
                    print(f"Skipping factor '{key}': No valid data after cleaning.")
                    continue
                    
                # Sort for merge_asof
                factor_data = factor_data.sort_values('dtime')
                
                print(f"Merging with {key} data using merge_asof (tolerance 5 mins)...")
                # Merge based on nearest preceding measurement within 5 minutes, per unit
                merged_data = pd.merge_asof(
                    merged_data, 
                    factor_data, 
                    on='dtime', 
                    by='unit_id',
                    direction='backward',
                    tolerance=pd.Timedelta('5minutes')
                )
                # Check if merge added the column
                if factor_metric_col in merged_data.columns:
                     factor_cols.append(factor_metric_col)
                     print(f"Successfully merged {key} ({factor_metric_col}).")
                else:
                     print(f"Warning: Merge with {key} did not add column {factor_metric_col}.")


            # Drop rows where merging failed for any factor used
            merged_data = merged_data.dropna(subset=['bytes_sec'] + factor_cols)
            
            if merged_data.empty or len(merged_data) < 10: # Need sufficient data for modeling
                print("Not enough merged data points to perform correlation/regression analysis.")
                return results
            
            print(f"Merged dataset size for factor analysis: {merged_data.shape}")
            
            results['merged_data_info'] = {'shape': merged_data.shape, 'columns': merged_data.columns.tolist()}

            # --- Correlation Analysis ---
            results['correlations'] = {}
            for factor_col in factor_cols:
                 try:
                    correlation = merged_data['bytes_sec'].corr(merged_data[factor_col])
                    results['correlations'][factor_col] = correlation
                    print(f"Correlation (bytes_sec vs {factor_col}): {correlation:.4f}")

                    # Scatter plot
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=merged_data, x=factor_col, y='bytes_sec', alpha=0.5)
                    # Add regression line for visualization
                    sns.regplot(data=merged_data, x=factor_col, y='bytes_sec', scatter=False, 
                                line_kws={"color":"red"})
                    plt.title(f'Impact of {factor_col} on Download Speed')
                    plt.xlabel(f'{factor_col}')
                    plt.ylabel('Download Speed (bytes/sec)')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(self.folders['correlation'], f'{factor_col}_download_correlation.png'))
                    plt.close()
                 except Exception as e:
                     print(f"Error during correlation/plotting for {factor_col}: {e}")

            # --- Multivariate Regression Analysis ---
            if len(factor_cols) > 0:
                print("Performing multivariate linear regression...")
                X = merged_data[factor_cols]
                y = merged_data['bytes_sec']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                if len(X_train) > len(factor_cols) and len(X_test) > 0: # Ensure enough samples
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test)
                    
                    results['multivariate_regression'] = {
                        'factors': factor_cols,
                        'coefficients': dict(zip(factor_cols, model.coef_)),
                        'intercept': model.intercept_,
                        'r2_score': r2_score(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                    print(f"Multivariate Regression R² score: {results['multivariate_regression']['r2_score']:.4f}")
                else:
                    print("Not enough data for train/test split in multivariate regression.")
            else:
                 print("No factors available for multivariate regression.")

        except Exception as e:
            print(f"An error occurred during factor analysis: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback

        # Zapisz wyniki do pliku
        report_file = os.path.join(self.folders['reports'], 'performance_factors_analysis.txt')
        with open(report_file, 'w') as f:
            f.write("# Analysis of Factors Affecting Download Performance\n\n")
            
            if 'merged_data_info' in results:
                 f.write(f"Merged Data Shape: {results['merged_data_info']['shape']}\n")
                 f.write(f"Merged Columns: {', '.join(results['merged_data_info']['columns'])}\n\n")
            
            if 'correlations' in results and results['correlations']:
                f.write("## Pearson Correlations with Download Speed (bytes_sec)\n\n")
                for factor, corr in results['correlations'].items():
                    f.write(f"- {factor}: {corr:.4f}\n")
            else: f.write("- No correlation results available.\n\n")

            if 'multivariate_regression' in results:
                 reg = results['multivariate_regression']
                 f.write("\n## Multivariate Regression Model (Predicting bytes_sec)\n\n")
                 f.write(f"Factors (Predictors): {', '.join(reg['factors'])}\n")
                 f.write("Coefficients:\n")
                 for factor, coef in reg['coefficients'].items():
                     f.write(f"- {factor}: {coef:.4f}\n")
                 f.write(f"Intercept: {reg['intercept']:.4f}\n")
                 f.write(f"R² score (Test): {reg['r2_score']:.4f}\n")
                 f.write(f"Mean Absolute Error (Test): {reg['mae']:.4f}\n")
                 f.write(f"Mean Squared Error (Test): {reg['mse']:.4f}\n")
                 f.write(f"Training Set Size: {reg['train_size']}\n")
                 f.write(f"Test Set Size: {reg['test_size']}\n\n")
                 
                 f.write("Interpretation (simplified):\n")
                 for factor, coef in reg['coefficients'].items():
                     change = "decreases" if coef < 0 else "increases"
                     f.write(f"  - A 1-unit increase in `{factor}` {change} download speed by approx. {abs(coef):.2f} bytes/sec, holding other factors constant.\n")
                 f.write("\n")
            else:
                 print("\nMultivariate regression model not built.")

        print(f"Factor analysis report saved to: {report_file}")
        return results

    def run_all_analyses(self):
        """
        Runs all implemented analyses and returns the collected results.
        
        Returns:
            dict: A dictionary containing results from all analyses.
                  Keys: 'summary', 'download_performance', 'upload_performance', 
                        'comparison', 'factor_analysis'
        """
        print("\n--- Running All Analyses ---")
        all_results = {}
        all_results['summary'] = self.generate_data_summary()
        all_results['download_performance'] = self.analyze_download_performance()
        all_results['upload_performance'] = self.analyze_upload_performance()
        all_results['comparison'] = self.analyze_download_upload_comparison()
        all_results['factor_analysis'] = self.analyze_performance_factors()
        print("\n--- All Analyses Complete ---")
        return all_results

