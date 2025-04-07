#!/usr/bin/env python3
"""
Moduł zawierający algorytmy do analizy danych.
Implementuje wszystkie kluczowe pojęcia z listy:
- Analiza danych (DA)
- Eksploracja danych (DE)
- Eksploracyjna analiza danych (EDA)
- Korelacja statystyczna
- Regresja liniowa
- Parametry ilościowe i jakościowe 
- Wartości odstające
- Kwartyle
- Rozstęp międzykwartylowy (IQR)
- Mapy ciepła
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class DataAnalyzer:
    """Klasa implementująca algorytmy analizy danych."""
    
    def __init__(self, data_path=None, output_dir="wyniki_analizy"):
        """
        Inicjalizacja analizatora danych.
        
        Args:
            data_path (str): Ścieżka do pliku z danymi (CSV)
            output_dir (str): Katalog wyjściowy dla wizualizacji
        """
        self.data_path = data_path
        self.data = None
        self.output_dir = output_dir
        
        # Utworzenie katalogu na wyniki
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_data(self, data_path=None):
        """
        Wczytanie danych z pliku CSV.
        
        Args:
            data_path (str, optional): Ścieżka do pliku CSV. Jeśli None, używa ścieżki z inicjalizacji.
        
        Returns:
            pandas.DataFrame: Wczytane dane
        """
        if data_path:
            self.data_path = data_path
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Kurwa, załadowano dane z {self.data_path}. Wymiary: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Kurwa, błąd podczas wczytywania danych: {e}")
            return None
    
    def data_exploration(self, save=True):
        """
        Eksploracja danych (DE) - podstawowe statystyki i informacje o danych.
        
        Args:
            save (bool): Czy zapisać wyniki do pliku
        
        Returns:
            dict: Słownik z podstawowymi statystykami
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return None
        
        # Podstawowe informacje
        info = {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.data.select_dtypes(exclude=[np.number]).columns)
        }
        
        # Statystyki dla kolumn numerycznych
        if info["numeric_columns"]:
            info["numeric_stats"] = self.data[info["numeric_columns"]].describe().to_dict()
        
        # Statystyki dla kolumn kategorycznych
        if info["categorical_columns"]:
            cat_stats = {}
            for col in info["categorical_columns"]:
                cat_stats[col] = self.data[col].value_counts().to_dict()
            info["categorical_stats"] = cat_stats
        
        if save:
            # Zapisz do pliku
            with open(os.path.join(self.output_dir, "data_exploration.txt"), "w") as f:
                f.write("# Eksploracja danych (DE)\n\n")
                f.write(f"Wymiary danych: {info['shape']}\n")
                f.write(f"Liczba kolumn: {len(info['columns'])}\n")
                f.write(f"Liczba wierszy: {info['shape'][0]}\n\n")
                
                f.write("## Kolumny\n")
                for col, dtype in info["dtypes"].items():
                    missing = info["missing_values"][col]
                    f.write(f"- {col} ({dtype}): {missing} brakujących wartości\n")
                
                f.write("\n## Statystyki dla kolumn numerycznych\n")
                if "numeric_stats" in info:
                    for col, stats in info["numeric_stats"].items():
                        f.write(f"\n### {col}\n")
                        for stat, value in stats.items():
                            f.write(f"- {stat}: {value}\n")
        
        return info
    
    def eda_visualizations(self, cols=None, save=True):
        """
        Eksploracyjna analiza danych (EDA) - tworzenie wizualizacji.
        
        Args:
            cols (list): Lista kolumn do analizy. Jeśli None, używa wszystkich kolumn numerycznych.
            save (bool): Czy zapisać wykresy do plików
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return
        
        # Jeśli nie podano kolumn, użyj wszystkich numerycznych
        if cols is None:
            cols = list(self.data.select_dtypes(include=[np.number]).columns)
        
        if not cols:
            print("Kurwa, brak kolumn numerycznych do analizy!")
            return
        
        # Utwórz podkatalog dla wizualizacji EDA
        eda_dir = os.path.join(self.output_dir, "eda_wizualizacje")
        if not os.path.exists(eda_dir) and save:
            os.makedirs(eda_dir)
        
        # 1. Histogramy
        for col in cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col].dropna(), kde=True)
            plt.title(f"Histogram dla {col}")
            plt.xlabel(col)
            plt.ylabel("Częstotliwość")
            if save:
                plt.savefig(os.path.join(eda_dir, f"histogram_{col}.png"))
                plt.close()
            else:
                plt.show()
        
        # 2. Boxploty (wykresy pudełkowe) - pokazują kwartyle i wartości odstające
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data[cols])
        plt.title("Boxplot dla kolumn numerycznych")
        plt.xticks(rotation=90)
        if save:
            plt.savefig(os.path.join(eda_dir, "boxplot_all.png"))
            plt.close()
        else:
            plt.show()
        
        # 3. Scatterplots (wykresy rozproszenia) dla par kolumn
        if len(cols) > 1:
            for i, col1 in enumerate(cols[:-1]):
                for col2 in cols[i+1:]:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=col1, y=col2, data=self.data)
                    plt.title(f"Scatter plot: {col1} vs {col2}")
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    if save:
                        plt.savefig(os.path.join(eda_dir, f"scatter_{col1}_vs_{col2}.png"))
                        plt.close()
                    else:
                        plt.show()
        
        # 4. Mapa ciepła korelacji
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data[cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='coolwarm', 
                    vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
        plt.title("Mapa ciepła korelacji")
        if save:
            plt.savefig(os.path.join(eda_dir, "correlation_heatmap.png"))
            plt.close()
        else:
            plt.show()
    
    def correlation_analysis(self, cols=None, method='pearson', save=True):
        """
        Analiza korelacji między zmiennymi.
        
        Args:
            cols (list): Lista kolumn do analizy korelacji. Jeśli None, używa wszystkich numerycznych.
            method (str): Metoda korelacji ('pearson', 'spearman', 'kendall')
            save (bool): Czy zapisać wyniki do pliku
        
        Returns:
            pandas.DataFrame: Macierz korelacji
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return None
        
        # Wybierz kolumny numeryczne, jeśli nie podano
        if cols is None:
            cols = list(self.data.select_dtypes(include=[np.number]).columns)
        
        if not cols:
            print("Kurwa, brak kolumn numerycznych do analizy korelacji!")
            return None
        
        # Oblicz macierz korelacji
        corr_matrix = self.data[cols].corr(method=method)
        
        if save:
            # Zapisz macierz korelacji do pliku CSV
            corr_matrix.to_csv(os.path.join(self.output_dir, f"correlation_matrix_{method}.csv"))
            
            # Wizualizacja macierzy korelacji
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', 
                        vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
            plt.title(f"Mapa ciepła korelacji ({method})")
            plt.savefig(os.path.join(self.output_dir, f"correlation_heatmap_{method}.png"))
            plt.close()
        
        return corr_matrix
    
    def linear_regression(self, x_col, y_col, test_size=0.3, save=True):
        """
        Przeprowadza analizę regresji liniowej.
        
        Args:
            x_col (str): Nazwa kolumny dla zmiennej niezależnej X
            y_col (str): Nazwa kolumny dla zmiennej zależnej Y
            test_size (float): Rozmiar zbioru testowego (0.0-1.0)
            save (bool): Czy zapisać wyniki i wykresy
        
        Returns:
            dict: Słownik z wynikami regresji (model, metryki)
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return None
        
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Kurwa, jedna z kolumn {x_col} lub {y_col} nie istnieje w danych!")
            return None
        
        # Przygotuj dane
        df = self.data[[x_col, y_col]].dropna()
        X = df[x_col].values.reshape(-1, 1)
        y = df[y_col].values
        
        # Podział na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Trenowanie modelu
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predykcja
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metryki modelu
        metrics = {
            'train': {
                'MAE': mean_absolute_error(y_train, y_train_pred),
                'MSE': mean_squared_error(y_train, y_train_pred),
                'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'R2': r2_score(y_train, y_train_pred)
            },
            'test': {
                'MAE': mean_absolute_error(y_test, y_test_pred),
                'MSE': mean_squared_error(y_test, y_test_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'R2': r2_score(y_test, y_test_pred)
            }
        }
        
        # SSE (Sum of Squared Errors)
        sse_train = np.sum((y_train - y_train_pred) ** 2)
        sse_test = np.sum((y_test - y_test_pred) ** 2)
        metrics['train']['SSE'] = sse_train
        metrics['test']['SSE'] = sse_test
        
        # Standard error of estimation
        n_train = len(y_train)
        n_test = len(y_test)
        std_error_train = np.sqrt(sse_train / (n_train - 2))
        std_error_test = np.sqrt(sse_test / (n_test - 2))
        metrics['train']['s'] = std_error_train
        metrics['test']['s'] = std_error_test
        
        # Parametry modelu
        params = {
            'intercept': model.intercept_,
            'coefficient': model.coef_[0],
            'equation': f"y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}"
        }
        
        results = {
            'model': model,
            'params': params,
            'metrics': metrics
        }
        
        if save:
            # Utwórz folder dla regresji
            reg_dir = os.path.join(self.output_dir, "regresja")
            if not os.path.exists(reg_dir):
                os.makedirs(reg_dir)
            
            # Zapisz wyniki do pliku
            with open(os.path.join(reg_dir, f"regresja_{x_col}_vs_{y_col}.txt"), "w") as f:
                f.write(f"# Regresja liniowa: {y_col} ~ {x_col}\n\n")
                f.write(f"Równanie: {params['equation']}\n\n")
                
                f.write("## Parametry modelu\n")
                f.write(f"Wyraz wolny (intercept): {params['intercept']:.6f}\n")
                f.write(f"Współczynnik kierunkowy: {params['coefficient']:.6f}\n\n")
                
                f.write("## Metryki dla zbioru treningowego\n")
                for metric, value in metrics['train'].items():
                    f.write(f"{metric}: {value:.6f}\n")
                
                f.write("\n## Metryki dla zbioru testowego\n")
                for metric, value in metrics['test'].items():
                    f.write(f"{metric}: {value:.6f}\n")
            
            # Wizualizacja regresji
            plt.figure(figsize=(12, 8))
            
            # Dane treningowe i testowe
            plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Dane treningowe')
            plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Dane testowe')
            
            # Linia regresji
            x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            plt.plot(x_range, y_pred, color='red', linewidth=2, label=params['equation'])
            
            plt.title(f"Regresja liniowa: {y_col} ~ {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(reg_dir, f"regresja_{x_col}_vs_{y_col}.png"))
            plt.close()
        
        return results
    
    def detect_outliers(self, cols=None, method='iqr', threshold=1.5, save=True):
        """
        Wykrywanie wartości odstających.
        
        Args:
            cols (list): Lista kolumn do analizy. Jeśli None, używa wszystkich numerycznych.
            method (str): Metoda wykrywania ('iqr', 'zscore')
            threshold (float): Próg dla IQR (typowo 1.5) lub Z-score (typowo 3)
            save (bool): Czy zapisać wyniki do pliku
        
        Returns:
            dict: Słownik z informacjami o wartościach odstających
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return None
        
        # Wybierz kolumny numeryczne, jeśli nie podano
        if cols is None:
            cols = list(self.data.select_dtypes(include=[np.number]).columns)
        
        if not cols:
            print("Kurwa, brak kolumn numerycznych do analizy wartości odstających!")
            return None
        
        outliers_info = {}
        
        for col in cols:
            # Pomiń kolumny z brakującymi danymi
            if self.data[col].isnull().all():
                continue
            
            if method == 'iqr':
                # Metoda IQR (Interquartile Range)
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.data[
                    (self.data[col] < lower_bound) | 
                    (self.data[col] > upper_bound)
                ]
                
                outliers_info[col] = {
                    'method': 'IQR',
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outliers_count': len(outliers),
                    'outliers_percentage': (len(outliers) / len(self.data)) * 100,
                    'outliers_indices': outliers.index.tolist()
                }
                
            elif method == 'zscore':
                # Metoda Z-score
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outliers_mask = z_scores > threshold
                outliers = self.data[col].iloc[outliers_mask]
                
                outliers_info[col] = {
                    'method': 'Z-score',
                    'threshold': threshold,
                    'outliers_count': np.sum(outliers_mask),
                    'outliers_percentage': (np.sum(outliers_mask) / len(z_scores)) * 100,
                    'outliers_indices': np.where(outliers_mask)[0].tolist()
                }
        
        if save:
            # Utwórz folder dla analiz wartości odstających
            outliers_dir = os.path.join(self.output_dir, "outliers")
            if not os.path.exists(outliers_dir):
                os.makedirs(outliers_dir)
            
            # Zapisz informacje o wartościach odstających
            with open(os.path.join(outliers_dir, f"outliers_{method}.txt"), "w") as f:
                f.write(f"# Analiza wartości odstających - metoda {method}\n\n")
                
                for col, info in outliers_info.items():
                    f.write(f"## Kolumna: {col}\n")
                    f.write(f"Metoda: {info['method']}\n")
                    
                    if method == 'iqr':
                        f.write(f"Q1: {info['Q1']:.6f}\n")
                        f.write(f"Q3: {info['Q3']:.6f}\n")
                        f.write(f"IQR: {info['IQR']:.6f}\n")
                        f.write(f"Dolna granica: {info['lower_bound']:.6f}\n")
                        f.write(f"Górna granica: {info['upper_bound']:.6f}\n")
                    else:
                        f.write(f"Próg Z-score: {info['threshold']:.6f}\n")
                    
                    f.write(f"Liczba wartości odstających: {info['outliers_count']}\n")
                    f.write(f"Procent wartości odstających: {info['outliers_percentage']:.2f}%\n\n")
            
            # Wizualizacja - boxplot z zaznaczonymi wartościami odstającymi
            for col in cols:
                if col in outliers_info:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x=self.data[col])
                    plt.title(f"Boxplot z wartościami odstającymi - {col}")
                    plt.xlabel(col)
                    plt.savefig(os.path.join(outliers_dir, f"boxplot_outliers_{col}.png"))
                    plt.close()
        
        return outliers_info
    
    def quartile_analysis(self, cols=None, save=True):
        """
        Analiza kwartylowa danych.
        
        Args:
            cols (list): Lista kolumn do analizy. Jeśli None, używa wszystkich numerycznych.
            save (bool): Czy zapisać wyniki do pliku
        
        Returns:
            dict: Słownik z informacjami o kwartylach
        """
        if self.data is None:
            print("Kurwa, brak danych do analizy! Najpierw wczytaj dane.")
            return None
        
        # Wybierz kolumny numeryczne, jeśli nie podano
        if cols is None:
            cols = list(self.data.select_dtypes(include=[np.number]).columns)
        
        if not cols:
            print("Kurwa, brak kolumn numerycznych do analizy kwartylowej!")
            return None
        
        quartile_info = {}
        
        for col in cols:
            # Pomiń kolumny z brakującymi danymi
            if self.data[col].isnull().all():
                continue
            
            # Oblicz kwartyle
            min_val = self.data[col].min()
            Q1 = self.data[col].quantile(0.25)
            median = self.data[col].quantile(0.5)
            Q3 = self.data[col].quantile(0.75)
            max_val = self.data[col].max()
            IQR = Q3 - Q1
            
            quartile_info[col] = {
                'min': min_val,
                'Q1': Q1,
                'median': median,
                'Q3': Q3,
                'max': max_val,
                'IQR': IQR,
                'range': max_val - min_val
            }
        
        if save:
            # Utwórz folder dla analiz kwartylowych
            quartile_dir = os.path.join(self.output_dir, "quartiles")
            if not os.path.exists(quartile_dir):
                os.makedirs(quartile_dir)
            
            # Zapisz informacje o kwartylach
            with open(os.path.join(quartile_dir, "quartile_analysis.txt"), "w") as f:
                f.write("# Analiza kwartylowa\n\n")
                
                for col, info in quartile_info.items():
                    f.write(f"## Kolumna: {col}\n")
                    f.write(f"Minimum: {info['min']:.6f}\n")
                    f.write(f"Q1 (25%): {info['Q1']:.6f}\n")
                    f.write(f"Mediana (50%): {info['median']:.6f}\n")
                    f.write(f"Q3 (75%): {info['Q3']:.6f}\n")
                    f.write(f"Maksimum: {info['max']:.6f}\n")
                    f.write(f"IQR (Q3-Q1): {info['IQR']:.6f}\n")
                    f.write(f"Rozstęp (max-min): {info['range']:.6f}\n\n")
            
            # Wizualizacja - boxplot
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=self.data[cols])
            plt.title("Boxplot dla kolumn numerycznych - analiza kwartylowa")
            plt.xticks(rotation=90)
            plt.savefig(os.path.join(quartile_dir, "boxplot_quartiles.png"))
            plt.close()
        
        return quartile_info
    
    def create_heatmap(self, data_matrix, columns=None, rows=None, title="Mapa ciepła", 
                       cmap="coolwarm", save=True, filename="heatmap.png"):
        """
        Tworzy mapę ciepła (heat map) z macierzy danych.
        
        Args:
            data_matrix (numpy.ndarray): Macierz danych do wizualizacji
            columns (list): Nazwy kolumn (opcjonalne)
            rows (list): Nazwy wierszy (opcjonalne)
            title (str): Tytuł mapy ciepła
            cmap (str): Paleta kolorów
            save (bool): Czy zapisać mapę do pliku
            filename (str): Nazwa pliku do zapisu
        """
        plt.figure(figsize=(12, 10))
        
        # Jeśli podano etykiety kolumn i wierszy, użyj ich
        if columns is not None and rows is not None:
            df = pd.DataFrame(data_matrix, index=rows, columns=columns)
            sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)
        else:
            sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5)
        
        plt.title(title)
        
        if save:
            heat_dir = os.path.join(self.output_dir, "heatmaps")
            if not os.path.exists(heat_dir):
                os.makedirs(heat_dir)
            
            plt.savefig(os.path.join(heat_dir, filename))
            plt.close()
        else:
            plt.show()


def main():
    """Przykładowe użycie modułu analizy danych."""
    # Przykład użycia z domyślnym plikiem CSV
    data_path = "dane/przyklad.csv"
    
    # Sprawdź czy plik istnieje, jeśli nie, wygeneruj przykładowe dane
    if not os.path.exists(data_path):
        print(f"Kurwa, plik {data_path} nie istnieje. Generuję przykładowe dane.")
        
        # Utwórz katalog na dane, jeśli nie istnieje
        if not os.path.exists("dane"):
            os.makedirs("dane")
        
        # Generowanie przykładowych danych
        np.random.seed(42)
        n_samples = 100
        
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(5, 2, n_samples)
        y = 3 * x1 + 2 * x2 + np.random.normal(0, 1, n_samples)
        
        categorical = np.random.choice(['A', 'B', 'C'], n_samples)
        
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'y': y,
            'category': categorical
        })
        
        df.to_csv(data_path, index=False)
        print(f"Kurwa, zapisano przykładowe dane do {data_path}")
    
    # Inicjalizacja analizatora
    analyzer = DataAnalyzer(data_path=data_path)
    
    # Wczytanie danych
    df = analyzer.load_data()
    
    if df is not None:
        # Eksploracja danych
        analyzer.data_exploration()
        
        # Wizualizacje EDA
        analyzer.eda_visualizations()
        
        # Analiza korelacji
        analyzer.correlation_analysis()
        
        # Regresja liniowa
        analyzer.linear_regression('x1', 'y')
        
        # Wykrywanie wartości odstających
        analyzer.detect_outliers()
        
        # Analiza kwartylowa
        analyzer.quartile_analysis()
        
        print("Kurwa, analiza zakończona! Wyniki zapisano w katalogu:", analyzer.output_dir)


if __name__ == "__main__":
    main() 