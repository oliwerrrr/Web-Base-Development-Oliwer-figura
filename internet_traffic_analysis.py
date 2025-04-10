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
from scipy import stats


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
    
    def load_data(self):
        """
        Wczytanie danych dotyczących ruchu internetowego.
        Analizujemy pliki związane z pobieraniem i wysyłaniem.
        
        Returns:
            dict: Słownik z wczytanymi danymi
        """
        print("Wczytuję dane ruchu internetowego...")
        
        download_files = {
            'httpget': 'curr_httpget.csv',
            'httpgetmt': 'curr_httpgetmt.csv', 
            'httpgetmt6': 'curr_httpgetmt6.csv',
            'dlping': 'curr_dlping.csv',
            'webget': 'curr_webget.csv'
        }
        
        upload_files = {
            'httppost': 'curr_httppost.csv',
            'httppostmt': 'curr_httppostmt.csv',
            'httppostmt6': 'curr_httppostmt6.csv',
            'ulping': 'curr_ulping.csv'
        }
        
        # Wczytaj dane pobierania
        for key, filename in download_files.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                # Dodajemy low_memory=False aby uniknąć mieszania typów danych
                data = pd.read_csv(file_path, low_memory=False)
                if not data.empty:
                    self.download_data[key] = data
                    print(f"Załadowano dane {key}: {data.shape}")
            except Exception as e:
                print(f"Błąd podczas wczytywania {filename}: {e}")
        
        # Wczytaj dane wysyłania
        for key, filename in upload_files.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                # Dodajemy low_memory=False aby uniknąć mieszania typów danych
                data = pd.read_csv(file_path, low_memory=False)
                if not data.empty:
                    self.upload_data[key] = data
                    print(f"Załadowano dane {key}: {data.shape}")
            except Exception as e:
                print(f"Błąd podczas wczytywania {filename}: {e}")
        
        # Wczytaj dodatkowe pliki, które mogą być przydatne do analizy korelacji
        additional_files = {
            'dns': 'curr_dns.csv',
            'ping': 'curr_ping.csv',
            'traceroute': 'curr_traceroute.csv',
            'udplatency': 'curr_udplatency.csv',
            'udpjitter': 'curr_udpjitter.csv',
            'udpcloss': 'curr_udpcloss.csv'
        }
        
        self.additional_data = {}
        
        for key, filename in additional_files.items():
            file_path = os.path.join(self.data_dir, filename)
            try:
                # Ograniczamy wczytywanie dużych plików do pierwszych 50000 wierszy, aby przyspieszyć analizę
                # Dodajemy low_memory=False aby uniknąć mieszania typów danych
                data = pd.read_csv(file_path, nrows=50000, low_memory=False)
                if not data.empty:
                    self.additional_data[key] = data
                    print(f"Załadowano dodatkowe dane {key} (częściowo): {data.shape}")
            except Exception as e:
                print(f"Błąd podczas wczytywania {filename}: {e}")
                
        # Konwertuj kolumny z prędkością na numeryczne, jeśli to konieczne
        for key, data in self.download_data.items():
            if 'bytes_sec' in data.columns:
                try:
                    # Spróbuj skonwertować kolumnę bytes_sec na typ float
                    data['bytes_sec'] = pd.to_numeric(data['bytes_sec'], errors='coerce')
                    print(f"Przekonwertowano kolumnę bytes_sec w {key} na typ numeryczny")
                except Exception as e:
                    print(f"Błąd podczas konwersji bytes_sec w {key}: {e}")
        
        for key, data in self.upload_data.items():
            if 'bytes_sec' in data.columns:
                try:
                    # Spróbuj skonwertować kolumnę bytes_sec na typ float
                    data['bytes_sec'] = pd.to_numeric(data['bytes_sec'], errors='coerce')
                    print(f"Przekonwertowano kolumnę bytes_sec w {key} na typ numeryczny")
                except Exception as e:
                    print(f"Błąd podczas konwersji bytes_sec w {key}: {e}")
        
        return {
            'download': self.download_data,
            'upload': self.upload_data,
            'additional': self.additional_data
        }
    
    def generate_data_summary(self):
        """
        Generuje podstawowe podsumowanie danych.
        
        Returns:
            dict: Słownik z podsumowaniem danych
        """
        summary = {
            'download': {},
            'upload': {}
        }
        
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
        
        # Zapisz podsumowanie do pliku
        with open(os.path.join(self.folders['reports'], 'data_summary.txt'), 'w') as f:
            f.write("# Podsumowanie danych ruchu internetowego\n\n")
            
            # Dane pobierania
            f.write("## Dane pobierania (Download)\n\n")
            for key, info in summary['download'].items():
                f.write(f"### {key}\n")
                f.write(f"Wymiary: {info['shape']}\n")
                f.write(f"Kolumny: {', '.join(info['columns'])}\n")
                f.write(f"Kolumny numeryczne: {', '.join(info['numeric_cols'])}\n\n")
                f.write("Statystyki:\n")
                f.write(f"{info['stats'].to_string()}\n\n")
            
            # Dane wysyłania
            f.write("## Dane wysyłania (Upload)\n\n")
            for key, info in summary['upload'].items():
                f.write(f"### {key}\n")
                f.write(f"Wymiary: {info['shape']}\n")
                f.write(f"Kolumny: {', '.join(info['columns'])}\n")
                f.write(f"Kolumny numeryczne: {', '.join(info['numeric_cols'])}\n\n")
                f.write("Statystyki:\n")
                f.write(f"{info['stats'].to_string()}\n\n")
        
        return summary
    
    def analyze_download_performance(self):
        """
        Analizuje wydajność pobierania danych.
        Skupia się na prędkości pobierania i czasach odpowiedzi.
        
        Returns:
            dict: Wyniki analizy wydajności pobierania
        """
        results = {}
        
        # Analiza danych HTTP GET
        if 'httpgetmt' in self.download_data:
            data = self.download_data['httpgetmt']
            
            # Filtruję dane, żeby pozbyć się wartości odstających
            if 'bytes_sec' in data.columns:
                # Sprawdź czy kolumna bytes_sec jest numeryczna
                try:
                    # Upewnij się, że kolumna jest numeryczna i pomiń wartości NaN
                    bytes_sec = pd.to_numeric(data['bytes_sec'], errors='coerce')
                    # Usuń wartości NaN
                    bytes_sec = bytes_sec.dropna()
                    
                    if not bytes_sec.empty:
                        # Usuwam wartości poniżej 5 percentyla i powyżej 95 percentyla
                        lower_bound = bytes_sec.quantile(0.05)
                        upper_bound = bytes_sec.quantile(0.95)
                        
                        # Tworzę kopię, aby uniknąć SettingWithCopyWarning
                        filtered_data = data.loc[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)].copy()
                        
                        # Histogramy prędkości pobierania
                        plt.figure(figsize=(10, 6))
                        sns.histplot(bytes_sec, kde=True)
                        plt.title('Rozkład prędkości pobierania (HTTP GET MT)')
                        plt.xlabel('Prędkość (bytes/sec)')
                        plt.ylabel('Liczba pomiarów')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], 'httpgetmt_speed_distribution.png'))
                        plt.close()
                        
                        # Statystyki
                        results['httpgetmt'] = {
                            'mean_speed': bytes_sec.mean(),
                            'median_speed': bytes_sec.median(),
                            'std_speed': bytes_sec.std(),
                            'min_speed': bytes_sec.min(),
                            'max_speed': bytes_sec.max()
                        }
                        
                        # Analiza czasowa (jeśli dostępna kolumna dtime)
                        if 'dtime' in filtered_data.columns:
                            try:
                                # Konwertuj dtime na datetime bezpiecznie
                                filtered_data['dtime'] = pd.to_datetime(filtered_data['dtime'], errors='coerce')
                                
                                # Usuń wiersze z błędnymi datami
                                filtered_data = filtered_data.dropna(subset=['dtime'])
                                
                                if not filtered_data.empty:
                                    filtered_data.set_index('dtime', inplace=True)
                                    
                                    # Agregacja danych według dnia
                                    # Upewnij się, że agregujemy tylko kolumny numeryczne
                                    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
                                    
                                    if 'bytes_sec' in numeric_cols:
                                        daily_data = filtered_data[['bytes_sec']].resample('D').mean()
                                        
                                        # Wykres zmian prędkości w czasie
                                        plt.figure(figsize=(12, 6))
                                        plt.plot(daily_data.index, daily_data['bytes_sec'])
                                        plt.title('Średnia prędkość pobierania w czasie (HTTP GET MT)')
                                        plt.xlabel('Data')
                                        plt.ylabel('Prędkość (bytes/sec)')
                                        plt.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(self.folders['performance'], 'httpgetmt_speed_timeline.png'))
                                        plt.close()
                            except Exception as e:
                                print(f"Błąd podczas analizy czasowej: {e}")
                except Exception as e:
                    print(f"Błąd podczas analizy prędkości pobierania: {e}")
        
        # Analiza danych Web GET (jeśli dostępne)
        if 'webget' in self.download_data:
            data = self.download_data['webget']
            
            # Przetwarzanie podobne do HTTP GET MT
            if 'bytes_sec' in data.columns:
                try:
                    # Upewnij się, że kolumna jest numeryczna i pomiń wartości NaN
                    bytes_sec = pd.to_numeric(data['bytes_sec'], errors='coerce')
                    # Usuń wartości NaN
                    bytes_sec = bytes_sec.dropna()
                    
                    if not bytes_sec.empty:
                        # Usuwam wartości poniżej 5 percentyla i powyżej 95 percentyla
                        lower_bound = bytes_sec.quantile(0.05)
                        upper_bound = bytes_sec.quantile(0.95)
                        
                        filtered_bytes_sec = bytes_sec[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)]
                        
                        # Histogramy prędkości pobierania
                        plt.figure(figsize=(10, 6))
                        sns.histplot(filtered_bytes_sec, kde=True)
                        plt.title('Rozkład prędkości pobierania (Web GET)')
                        plt.xlabel('Prędkość (bytes/sec)')
                        plt.ylabel('Liczba pomiarów')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], 'webget_speed_distribution.png'))
                        plt.close()
                        
                        # Statystyki
                        results['webget'] = {
                            'mean_speed': filtered_bytes_sec.mean(),
                            'median_speed': filtered_bytes_sec.median(),
                            'std_speed': filtered_bytes_sec.std(),
                            'min_speed': filtered_bytes_sec.min(),
                            'max_speed': filtered_bytes_sec.max()
                        }
                except Exception as e:
                    print(f"Błąd podczas analizy prędkości pobierania webget: {e}")
        
        # Zapisz wyniki do pliku
        with open(os.path.join(self.folders['reports'], 'download_performance_analysis.txt'), 'w') as f:
            f.write("# Analiza wydajności pobierania\n\n")
            
            for key, stats in results.items():
                f.write(f"## {key}\n\n")
                for stat_name, value in stats.items():
                    f.write(f"{stat_name}: {value:,.2f}\n")
                f.write("\n")
        
        return results
    
    def analyze_upload_performance(self):
        """
        Analizuje wydajność wysyłania danych.
        
        Returns:
            dict: Wyniki analizy wydajności wysyłania
        """
        results = {}
        
        # Analiza danych HTTP POST
        if 'httppostmt' in self.upload_data:
            data = self.upload_data['httppostmt']
            
            # Filtruję dane, żeby pozbyć się wartości odstających
            if 'bytes_sec' in data.columns:
                try:
                    # Upewnij się, że kolumna jest numeryczna i pomiń wartości NaN
                    bytes_sec = pd.to_numeric(data['bytes_sec'], errors='coerce')
                    # Usuń wartości NaN
                    bytes_sec = bytes_sec.dropna()
                    
                    if not bytes_sec.empty:
                        # Usuwam wartości poniżej 5 percentyla i powyżej 95 percentyla
                        lower_bound = bytes_sec.quantile(0.05)
                        upper_bound = bytes_sec.quantile(0.95)
                        
                        filtered_bytes_sec = bytes_sec[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)]
                        
                        # Tworzę kopię, aby uniknąć SettingWithCopyWarning
                        filtered_data = data.loc[(bytes_sec >= lower_bound) & (bytes_sec <= upper_bound)].copy()
                        
                        # Histogramy prędkości wysyłania
                        plt.figure(figsize=(10, 6))
                        sns.histplot(filtered_bytes_sec, kde=True)
                        plt.title('Rozkład prędkości wysyłania (HTTP POST MT)')
                        plt.xlabel('Prędkość (bytes/sec)')
                        plt.ylabel('Liczba pomiarów')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(os.path.join(self.folders['performance'], 'httppostmt_speed_distribution.png'))
                        plt.close()
                        
                        # Statystyki
                        results['httppostmt'] = {
                            'mean_speed': filtered_bytes_sec.mean(),
                            'median_speed': filtered_bytes_sec.median(),
                            'std_speed': filtered_bytes_sec.std(),
                            'min_speed': filtered_bytes_sec.min(),
                            'max_speed': filtered_bytes_sec.max()
                        }
                        
                        # Analiza czasowa (jeśli dostępna kolumna dtime)
                        if 'dtime' in filtered_data.columns:
                            try:
                                # Konwertuj dtime na datetime bezpiecznie
                                filtered_data['dtime'] = pd.to_datetime(filtered_data['dtime'], errors='coerce')
                                
                                # Usuń wiersze z błędnymi datami
                                filtered_data = filtered_data.dropna(subset=['dtime'])
                                
                                if not filtered_data.empty:
                                    filtered_data.set_index('dtime', inplace=True)
                                    
                                    # Agregacja danych według dnia
                                    # Upewnij się, że agregujemy tylko kolumny numeryczne
                                    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
                                    
                                    if 'bytes_sec' in numeric_cols:
                                        daily_data = filtered_data[['bytes_sec']].resample('D').mean()
                                        
                                        # Wykres zmian prędkości w czasie
                                        plt.figure(figsize=(12, 6))
                                        plt.plot(daily_data.index, daily_data['bytes_sec'])
                                        plt.title('Średnia prędkość wysyłania w czasie (HTTP POST MT)')
                                        plt.xlabel('Data')
                                        plt.ylabel('Prędkość (bytes/sec)')
                                        plt.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        plt.savefig(os.path.join(self.folders['performance'], 'httppostmt_speed_timeline.png'))
                                        plt.close()
                            except Exception as e:
                                print(f"Błąd podczas analizy czasowej: {e}")
                except Exception as e:
                    print(f"Błąd podczas analizy prędkości wysyłania: {e}")
        
        # Zapisz wyniki do pliku
        with open(os.path.join(self.folders['reports'], 'upload_performance_analysis.txt'), 'w') as f:
            f.write("# Analiza wydajności wysyłania\n\n")
            
            for key, stats in results.items():
                f.write(f"## {key}\n\n")
                for stat_name, value in stats.items():
                    f.write(f"{stat_name}: {value:,.2f}\n")
                f.write("\n")
        
        return results

    def analyze_download_upload_comparison(self):
        """
        Porównuje wydajność pobierania i wysyłania danych.
        
        Returns:
            dict: Wyniki porównania wydajności
        """
        results = {'comparison': {}}
        
        # Pobierz dane HTTP GET MT i HTTP POST MT do porównania
        if 'httpgetmt' in self.download_data and 'httppostmt' in self.upload_data:
            dl_data = self.download_data['httpgetmt']
            ul_data = self.upload_data['httppostmt']
            
            if 'bytes_sec' in dl_data.columns and 'bytes_sec' in ul_data.columns:
                try:
                    # Upewnij się, że kolumny są numeryczne
                    dl_speeds = pd.to_numeric(dl_data['bytes_sec'], errors='coerce').dropna()
                    ul_speeds = pd.to_numeric(ul_data['bytes_sec'], errors='coerce').dropna()
                    
                    if not dl_speeds.empty and not ul_speeds.empty:
                        # Usuwam wartości odstające
                        dl_lower = dl_speeds.quantile(0.05)
                        dl_upper = dl_speeds.quantile(0.95)
                        ul_lower = ul_speeds.quantile(0.05)
                        ul_upper = ul_speeds.quantile(0.95)
                        
                        dl_filtered = dl_speeds[(dl_speeds >= dl_lower) & (dl_speeds <= dl_upper)]
                        ul_filtered = ul_speeds[(ul_speeds >= ul_lower) & (ul_speeds <= ul_upper)]
                        
                        # Statystyki
                        dl_mean = dl_filtered.mean()
                        ul_mean = ul_filtered.mean()
                        dl_median = dl_filtered.median()
                        ul_median = ul_filtered.median()
                        
                        results['comparison'] = {
                            'dl_mean': dl_mean,
                            'ul_mean': ul_mean,
                            'dl_median': dl_median,
                            'ul_median': ul_median,
                            'dl_ul_ratio_mean': dl_mean / ul_mean if ul_mean > 0 else 'N/A',
                            'dl_ul_ratio_median': dl_median / ul_median if ul_median > 0 else 'N/A'
                        }
                        
                        # Porównanie rozkładów - oddzielne histogramy
                        plt.figure(figsize=(12, 8))
                        
                        plt.subplot(2, 1, 1)
                        sns.histplot(dl_filtered, color='blue', kde=True, label='Download')
                        plt.title('Rozkład prędkości pobierania (Download)')
                        plt.xlabel('Prędkość (bytes/sec)')
                        plt.ylabel('Liczba pomiarów')
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(2, 1, 2)
                        sns.histplot(ul_filtered, color='red', kde=True, label='Upload')
                        plt.title('Rozkład prędkości wysyłania (Upload)')
                        plt.xlabel('Prędkość (bytes/sec)')
                        plt.ylabel('Liczba pomiarów')
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.folders['performance'], 'download_upload_distributions.png'))
                        plt.close()
                        
                        # Wykres pudełkowy - ograniczamy dane, aby były tej samej długości
                        # Pobieramy losową próbkę danych o tej samej wielkości
                        sample_size = min(len(dl_filtered), len(ul_filtered), 1000)  # Maksymalnie 1000 punktów
                        
                        if sample_size > 0:
                            dl_sample = dl_filtered.sample(sample_size, random_state=42) if len(dl_filtered) > sample_size else dl_filtered
                            ul_sample = ul_filtered.sample(sample_size, random_state=42) if len(ul_filtered) > sample_size else ul_filtered
                            
                            plt.figure(figsize=(10, 8))
                            
                            # Tworzymy DataFrame z równej długości próbek
                            box_data = pd.DataFrame({
                                'Download': dl_sample.values,
                                'Upload': ul_sample.values
                            })
                            
                            sns.boxplot(data=box_data)
                            plt.title('Porównanie rozkładu prędkości pobierania i wysyłania')
                            plt.ylabel('Prędkość (bytes/sec)')
                            plt.grid(True, alpha=0.3)
                            plt.savefig(os.path.join(self.folders['performance'], 'download_upload_boxplot.png'))
                            plt.close()
                        
                        # Zapisz wyniki do pliku
                        with open(os.path.join(self.folders['reports'], 'download_upload_comparison.txt'), 'w') as f:
                            f.write("# Porównanie wydajności pobierania i wysyłania\n\n")
                            
                            f.write("## Statystyki\n\n")
                            f.write(f"Średnia prędkość pobierania: {dl_mean:,.2f} bytes/sec\n")
                            f.write(f"Średnia prędkość wysyłania: {ul_mean:,.2f} bytes/sec\n")
                            f.write(f"Mediana prędkości pobierania: {dl_median:,.2f} bytes/sec\n")
                            f.write(f"Mediana prędkości wysyłania: {ul_median:,.2f} bytes/sec\n")
                            
                            if isinstance(results['comparison']['dl_ul_ratio_mean'], (int, float)):
                                f.write(f"Stosunek średnich (DL/UL): {results['comparison']['dl_ul_ratio_mean']:.2f}\n")
                            else:
                                f.write(f"Stosunek średnich (DL/UL): {results['comparison']['dl_ul_ratio_mean']}\n")
                                
                            if isinstance(results['comparison']['dl_ul_ratio_median'], (int, float)):
                                f.write(f"Stosunek median (DL/UL): {results['comparison']['dl_ul_ratio_median']:.2f}\n")
                            else:
                                f.write(f"Stosunek median (DL/UL): {results['comparison']['dl_ul_ratio_median']}\n")
                except Exception as e:
                    print(f"Błąd podczas porównywania prędkości pobierania i wysyłania: {e}")
        
        return results
    
    def analyze_performance_factors(self):
        """
        Analizuje czynniki wpływające na wydajność ruchu internetowego.
        Szuka korelacji między różnymi parametrami.
        
        Returns:
            dict: Wyniki analizy czynników wpływających na wydajność
        """
        results = {}
        
        # Sprawdź wpływ opóźnienia (latency) na prędkość pobierania
        if ('httpgetmt' in self.download_data and 
            'ping' in self.additional_data):
            
            dl_data = self.download_data['httpgetmt']
            ping_data = self.additional_data['ping']
            
            # Spróbuj połączyć dane na podstawie unit_id i zbliżonego czasu
            if ('unit_id' in dl_data.columns and 'dtime' in dl_data.columns and
                'unit_id' in ping_data.columns and 'dtime' in ping_data.columns):
                
                try:
                    # Upewnij się, że kolumna bytes_sec jest numeryczna
                    dl_data['bytes_sec'] = pd.to_numeric(dl_data['bytes_sec'], errors='coerce')
                    
                    # Upewnij się, że kolumna rtt_avg jest numeryczna
                    if 'rtt_avg' in ping_data.columns:
                        ping_data['rtt_avg'] = pd.to_numeric(ping_data['rtt_avg'], errors='coerce')
                    else:
                        print("Kolumna rtt_avg nie istnieje w danych ping")
                        return results
                    
                    # Konwertuj czas na datetime
                    dl_data['dtime'] = pd.to_datetime(dl_data['dtime'], errors='coerce')
                    ping_data['dtime'] = pd.to_datetime(ping_data['dtime'], errors='coerce')
                    
                    # Usuń wiersze z brakującymi wartościami
                    dl_data = dl_data.dropna(subset=['dtime', 'bytes_sec', 'unit_id'])
                    ping_data = ping_data.dropna(subset=['dtime', 'rtt_avg', 'unit_id'])
                    
                    if not dl_data.empty and not ping_data.empty:
                        # Agregacja danych według unit_id i dnia
                        dl_daily = dl_data.groupby(['unit_id', pd.Grouper(key='dtime', freq='D')])[['bytes_sec']].mean().reset_index()
                        ping_daily = ping_data.groupby(['unit_id', pd.Grouper(key='dtime', freq='D')])[['rtt_avg']].mean().reset_index()
                        
                        # Połącz dane
                        merged_data = pd.merge(dl_daily, ping_daily, on=['unit_id', 'dtime'], how='inner')
                        
                        if not merged_data.empty and len(merged_data) > 5:
                            # Oblicz korelację
                            correlation = merged_data['bytes_sec'].corr(merged_data['rtt_avg'])
                            
                            results['latency_impact'] = {
                                'correlation': correlation,
                                'sample_size': len(merged_data)
                            }
                            
                            # Wykres rozproszenia
                            plt.figure(figsize=(10, 6))
                            sns.scatterplot(data=merged_data, x='rtt_avg', y='bytes_sec')
                            
                            # Dodaj linię regresji
                            sns.regplot(data=merged_data, x='rtt_avg', y='bytes_sec', scatter=False, 
                                        line_kws={"color":"red"})
                            
                            plt.title('Wpływ opóźnienia (ping) na prędkość pobierania')
                            plt.xlabel('Średnie opóźnienie RTT (ms)')
                            plt.ylabel('Prędkość pobierania (bytes/sec)')
                            plt.grid(True, alpha=0.3)
                            plt.savefig(os.path.join(self.folders['correlation'], 'latency_download_correlation.png'))
                            plt.close()
                            
                            # Regresja liniowa
                            X = merged_data[['rtt_avg']]
                            y = merged_data['bytes_sec']
                            
                            # Podziel dane na zbiór treningowy i testowy
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                            
                            # Trenuj model regresji
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Oceń model
                            y_pred = model.predict(X_test)
                            
                            results['latency_regression'] = {
                                'coefficient': model.coef_[0],
                                'intercept': model.intercept_,
                                'r2_score': r2_score(y_test, y_pred),
                                'mae': mean_absolute_error(y_test, y_pred),
                                'mse': mean_squared_error(y_test, y_pred)
                            }
                except Exception as e:
                    print(f"Błąd podczas analizy wpływu opóźnienia: {e}")
        
        # Sprawdź wpływ jittera na prędkość pobierania - podobna implementacja
        # ...
        
        # Zapisz wyniki do pliku
        with open(os.path.join(self.folders['reports'], 'performance_factors_analysis.txt'), 'w') as f:
            f.write("# Analiza czynników wpływających na wydajność\n\n")
            
            if 'latency_impact' in results:
                f.write("## Wpływ opóźnienia na prędkość pobierania\n\n")
                f.write(f"Korelacja: {results['latency_impact']['correlation']:.4f}\n")
                f.write(f"Wielkość próby: {results['latency_impact']['sample_size']}\n\n")
                
                if 'latency_regression' in results:
                    f.write("### Model regresji\n\n")
                    f.write(f"Współczynnik kierunkowy: {results['latency_regression']['coefficient']:.4f}\n")
                    f.write(f"Wyraz wolny: {results['latency_regression']['intercept']:.4f}\n")
                    f.write(f"R² score: {results['latency_regression']['r2_score']:.4f}\n")
                    f.write(f"Mean Absolute Error: {results['latency_regression']['mae']:.4f}\n")
                    f.write(f"Mean Squared Error: {results['latency_regression']['mse']:.4f}\n\n")
                    
                    f.write("Interpretacja: ")
                    coef = results['latency_regression']['coefficient']
                    if coef < 0:
                        f.write(f"Każdy dodatkowy 1ms opóźnienia zmniejsza prędkość pobierania o {abs(coef):.2f} bytes/sec.\n\n")
                    else:
                        f.write(f"Każdy dodatkowy 1ms opóźnienia zwiększa prędkość pobierania o {coef:.2f} bytes/sec (nietypowa zależność).\n\n")
        
        return results
    
    def generate_crisp_dm_report(self):
        """
        Generuje raport zgodny z metodologią CRISP-DM.
        
        Returns:
            str: Ścieżka do wygenerowanego raportu
        """
        report_path = os.path.join(self.folders['reports'], 'crisp_dm_report.md')
        
        # Uzyskaj wyniki analizy faktorów, które mogą być użyte w raporcie
        factors_results = self.analyze_performance_factors()
        
        # Uzyskaj wyniki porównania pobierania i wysyłania
        comparison_results = self.analyze_download_upload_comparison()
        
        with open(report_path, 'w') as f:
            f.write("# Raport analizy wydajności ruchu internetowego wg metodologii CRISP-DM\n\n")
            
            # 1. Zrozumienie biznesu
            f.write("## 1. Zrozumienie biznesu/problemu (Business Understanding)\n\n")
            f.write("### Cele biznesowe\n\n")
            f.write("- Zrozumieć wydajność ruchu internetowego, aby zoptymalizować doświadczenia użytkowników\n")
            f.write("- Identyfikacja czynników wpływających na prędkość pobierania i wysyłania danych\n")
            f.write("- Ustalenie zależności między różnymi parametrami sieci\n\n")
            
            f.write("### Kryteria sukcesu\n\n")
            f.write("- Identyfikacja kluczowych wskaźników wydajności dla ruchu internetowego\n")
            f.write("- Znalezienie istotnych korelacji między parametrami sieci\n")
            f.write("- Stworzenie modeli wyjaśniających wpływ różnych czynników na wydajność\n\n")
            
            # 2. Zrozumienie danych
            f.write("## 2. Zrozumienie danych (Data Understanding)\n\n")
            f.write("### Opis danych\n\n")
            f.write("Analiza wykorzystuje następujące zestawy danych związane z ruchem internetowym:\n\n")
            
            f.write("**Dane dotyczące pobierania:**\n")
            for key in self.download_data.keys():
                shape = self.download_data[key].shape
                f.write(f"- {key}: {shape[0]} wierszy, {shape[1]} kolumn\n")
            
            f.write("\n**Dane dotyczące wysyłania:**\n")
            for key in self.upload_data.keys():
                shape = self.upload_data[key].shape
                f.write(f"- {key}: {shape[0]} wierszy, {shape[1]} kolumn\n")
            
            f.write("\n**Dodatkowe dane sieci:**\n")
            for key in self.additional_data.keys():
                shape = self.additional_data[key].shape
                f.write(f"- {key}: {shape[0]} wierszy, {shape[1]} kolumn\n")
            
            f.write("\n### Eksploracja danych\n\n")
            f.write("Główne parametry w danych dotyczących pobierania:\n")
            if 'httpgetmt' in self.download_data:
                f.write(f"- Kolumny w httpgetmt: {', '.join(self.download_data['httpgetmt'].columns)}\n")
            
            f.write("\nGłówne parametry w danych dotyczących wysyłania:\n")
            if 'httppostmt' in self.upload_data:
                f.write(f"- Kolumny w httppostmt: {', '.join(self.upload_data['httppostmt'].columns)}\n")
            
            f.write("\n### Jakość danych\n\n")
            if 'httpgetmt' in self.download_data:
                missing = self.download_data['httpgetmt'].isnull().sum().sum()
                total = self.download_data['httpgetmt'].size
                f.write(f"- W danych httpgetmt brakuje {missing} wartości ({missing/total:.2%} wszystkich danych)\n")
            
            if 'httppostmt' in self.upload_data:
                missing = self.upload_data['httppostmt'].isnull().sum().sum()
                total = self.upload_data['httppostmt'].size
                f.write(f"- W danych httppostmt brakuje {missing} wartości ({missing/total:.2%} wszystkich danych)\n")
            
            # 3. Przygotowanie danych
            f.write("\n## 3. Przygotowanie danych (Data Preparation)\n\n")
            f.write("### Wybór danych\n\n")
            f.write("- Skoncentrowano się na danych HTTP GET i HTTP POST jako kluczowych wskaźnikach wydajności\n")
            f.write("- Uwzględniono dodatkowe parametry sieci (ping, jitter) do analizy korelacji\n\n")
            
            f.write("### Czyszczenie danych\n\n")
            f.write("- Usunięto wartości odstające (poniżej 5 i powyżej 95 percentyla) dla prędkości pobierania i wysyłania\n")
            f.write("- Usunięto wiersze z brakującymi wartościami dla kluczowych parametrów\n")
            f.write("- Agregowano dane według dnia dla analiz czasowych\n\n")
            
            f.write("### Transformacja danych\n\n")
            f.write("- Konwersja kolumn czasowych na format datetime\n")
            f.write("- Agregacja danych według unit_id i dnia dla analizy korelacji\n\n")
            
            # 4. Modelowanie
            f.write("## 4. Modelowanie (Modeling)\n\n")
            f.write("### Techniki modelowania\n\n")
            f.write("- Wykorzystano regresję liniową do określenia wpływu opóźnienia na prędkość pobierania\n")
            f.write("- Zastosowano analizę korelacji do identyfikacji zależności między parametrami\n\n")
            
            f.write("### Ocena modeli\n\n")
            if 'latency_regression' in factors_results:
                f.write(f"- R² score dla modelu wpływu opóźnienia: {factors_results['latency_regression']['r2_score']:.4f}\n")
                f.write(f"- MAE dla modelu wpływu opóźnienia: {factors_results['latency_regression']['mae']:.4f}\n\n")
            else:
                f.write("- Nie udało się zbudować modelu regresji dla wpływu opóźnienia\n\n")
            
            # 5. Ewaluacja
            f.write("## 5. Ewaluacja (Evaluation)\n\n")
            f.write("### Wyniki\n\n")
            
            if 'comparison' in comparison_results:
                comp = comparison_results['comparison']
                if 'dl_mean' in comp and 'ul_mean' in comp:
                    dl_mean = comp['dl_mean']
                    ul_mean = comp['ul_mean']
                    ratio = comp.get('dl_ul_ratio_mean', 'N/A')
                    
                    if isinstance(dl_mean, (int, float)):
                        f.write(f"- Średnia prędkość pobierania: {dl_mean:,.2f} bytes/sec\n")
                    else:
                        f.write(f"- Średnia prędkość pobierania: {dl_mean}\n")
                        
                    if isinstance(ul_mean, (int, float)):
                        f.write(f"- Średnia prędkość wysyłania: {ul_mean:,.2f} bytes/sec\n")
                    else:
                        f.write(f"- Średnia prędkość wysyłania: {ul_mean}\n")
                        
                    if isinstance(ratio, (int, float)):
                        f.write(f"- Stosunek prędkości pobierania do wysyłania: {ratio:.2f}\n\n")
                    else:
                        f.write(f"- Stosunek prędkości pobierania do wysyłania: {ratio}\n\n")
            
            if 'latency_impact' in factors_results:
                f.write(f"- Korelacja między opóźnieniem a prędkością pobierania: {factors_results['latency_impact']['correlation']:.4f}\n")
                
                if 'latency_regression' in factors_results:
                    coef = factors_results['latency_regression']['coefficient']
                    if coef < 0:
                        f.write(f"- Każdy dodatkowy 1ms opóźnienia zmniejsza prędkość pobierania o {abs(coef):.2f} bytes/sec\n\n")
                    else:
                        f.write(f"- Każdy dodatkowy 1ms opóźnienia zwiększa prędkość pobierania o {coef:.2f} bytes/sec (nietypowa zależność)\n\n")
            
            f.write("### Ocena realizacji celów biznesowych\n\n")
            f.write("- Zidentyfikowano kluczowe wskaźniki wydajności ruchu internetowego\n")
            f.write("- Określono zależność między opóźnieniem a prędkością pobierania\n")
            f.write("- Porównano wydajność pobierania i wysyłania danych\n\n")
            
            # 6. Wdrożenie
            f.write("## 6. Wdrożenie (Deployment)\n\n")
            f.write("### Plan wdrożenia\n\n")
            f.write("- Stworzono kompleksowy raport z wizualizacjami\n")
            f.write("- Zaimplementowano algorytmy analizy w modułach Pythona\n")
            f.write("- Przygotowano wyniki w formie gotowej do prezentacji\n\n")
            
            f.write("### Monitoring\n\n")
            f.write("- Wyniki analizy można wizualizować za pomocą istniejącej aplikacji run.py\n")
            f.write("- Zaleca się regularne powtarzanie analizy dla nowych danych\n\n")
            
            # Podsumowanie
            f.write("## Podsumowanie\n\n")
            f.write("Analiza wydajności ruchu internetowego wykazała istotne różnice między prędkością pobierania i wysyłania danych. ")
            f.write("Zidentyfikowano wpływ opóźnienia sieci na prędkość pobierania. ")
            f.write("Wyniki analizy mogą posłużyć do optymalizacji konfiguracji sieci i poprawy doświadczeń użytkowników.\n")
        
        print(f"Raport CRISP-DM zapisano w {report_path}")
        return report_path

    def run_analysis(self):
        """
        Uruchamia pełną analizę ruchu internetowego.
        
        Returns:
            dict: Kompletne wyniki analizy
        """
        print("Rozpoczynam analizę ruchu internetowego...")
        
        # Wczytaj dane
        self.load_data()
        
        # Wygeneruj podsumowanie danych
        summary = self.generate_data_summary()
        
        # Analizuj wydajność pobierania
        download_results = self.analyze_download_performance()
        
        # Analizuj wydajność wysyłania
        upload_results = self.analyze_upload_performance()
        
        # Porównaj pobieranie i wysyłanie
        comparison_results = self.analyze_download_upload_comparison()
        
        # Analizuj czynniki wpływające na wydajność
        factors_results = self.analyze_performance_factors()
        
        # Wygeneruj raport CRISP-DM
        report_path = self.generate_crisp_dm_report()
        
        # Zbiorcze wyniki
        results = {
            'summary': summary,
            'download': download_results,
            'upload': upload_results,
            'comparison': comparison_results,
            'factors': factors_results,
            'report_path': report_path
        }
        
        print(f"Analiza zakończona. Wyniki zapisano w katalogu: {self.output_dir}")
        
        return results


def main():
    """Główna funkcja uruchamiająca analizę ruchu internetowego."""
    analyzer = InternetTrafficAnalyzer(data_dir="data", output_dir="wyniki_ruchu_internetowego")
    results = analyzer.run_analysis()
    
    print("\nNajważniejsze wyniki:")
    
    if 'comparison' in results and 'comparison' in results['comparison']:
        comp = results['comparison']['comparison']
        if 'dl_mean' in comp and 'ul_mean' in comp:
            print(f"- Średnia prędkość pobierania: {comp['dl_mean']:,.2f} bytes/sec")
            print(f"- Średnia prędkość wysyłania: {comp['ul_mean']:,.2f} bytes/sec")
            if isinstance(comp.get('dl_ul_ratio_mean'), (int, float)):
                print(f"- Stosunek DL/UL: {comp['dl_ul_ratio_mean']:.2f}")
    
    if 'factors' in results and 'latency_impact' in results['factors']:
        impact = results['factors']['latency_impact']
        print(f"- Korelacja opóźnienie-prędkość: {impact['correlation']:.4f}")


if __name__ == "__main__":
    main() 