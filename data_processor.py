#!/usr/bin/env python3
"""
Moduł do przetwarzania danych przed analizą.
Implementuje funkcje do czyszczenia, transformacji i przygotowania danych.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


class DataProcessor:
    """Klasa implementująca metody przetwarzania danych."""
    
    def __init__(self, data=None):
        """
        Inicjalizacja procesora danych.
        
        Args:
            data (pandas.DataFrame, optional): Dane do przetworzenia
        """
        self.data = data
        self.transformations = []
        self.original_data = None
        self.numeric_columns = None
        self.categorical_columns = None
    
    def load_data(self, file_path):
        """
        Wczytuje dane z pliku CSV.
        
        Args:
            file_path (str): Ścieżka do pliku CSV
        
        Returns:
            pandas.DataFrame: Wczytane dane
        """
        try:
            self.data = pd.read_csv(file_path)
            self.original_data = self.data.copy()
            self._identify_column_types()
            print(f"Kurwa, załadowano dane z {file_path}. Wymiary: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Kurwa, błąd podczas wczytywania danych: {e}")
            return None
    
    def _identify_column_types(self):
        """Identyfikuje typy kolumn w danych."""
        if self.data is None:
            print("Kurwa, brak danych do analizy!")
            return
        
        self.numeric_columns = list(self.data.select_dtypes(include=[np.number]).columns)
        self.categorical_columns = list(self.data.select_dtypes(exclude=[np.number]).columns)
        
        print(f"Kolumny numeryczne: {self.numeric_columns}")
        print(f"Kolumny kategoryczne: {self.categorical_columns}")
    
    def remove_duplicates(self):
        """
        Usuwa duplikaty z danych.
        
        Returns:
            pandas.DataFrame: Dane bez duplikatów
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)
        
        self.transformations.append(f"Usunięto {removed_rows} duplikatów")
        print(f"Kurwa, usunięto {removed_rows} duplikatów")
        
        return self.data
    
    def handle_missing_values(self, strategy='mean', categorical_strategy='most_frequent'):
        """
        Obsługuje brakujące wartości w danych.
        
        Args:
            strategy (str): Strategia dla danych numerycznych ('mean', 'median', 'most_frequent', 'constant')
            categorical_strategy (str): Strategia dla danych kategorycznych ('most_frequent', 'constant')
        
        Returns:
            pandas.DataFrame: Dane z uzupełnionymi wartościami
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        # Obsługa brakujących wartości dla kolumn numerycznych
        if self.numeric_columns:
            imputer = SimpleImputer(strategy=strategy)
            self.data[self.numeric_columns] = imputer.fit_transform(self.data[self.numeric_columns])
            self.transformations.append(f"Uzupełniono brakujące wartości numeryczne strategią '{strategy}'")
            print(f"Kurwa, uzupełniono brakujące wartości numeryczne strategią '{strategy}'")
        
        # Obsługa brakujących wartości dla kolumn kategorycznych
        if self.categorical_columns:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            self.data[self.categorical_columns] = cat_imputer.fit_transform(self.data[self.categorical_columns])
            self.transformations.append(f"Uzupełniono brakujące wartości kategoryczne strategią '{categorical_strategy}'")
            print(f"Kurwa, uzupełniono brakujące wartości kategoryczne strategią '{categorical_strategy}'")
        
        return self.data
    
    def scale_features(self, method='standard', columns=None):
        """
        Skaluje cechy numeryczne.
        
        Args:
            method (str): Metoda skalowania ('standard', 'minmax')
            columns (list, optional): Lista kolumn do skalowania. Jeśli None, używa wszystkich numerycznych.
        
        Returns:
            pandas.DataFrame: Dane ze skalowanymi cechami
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("Kurwa, brak kolumn numerycznych do skalowania!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if not existing_columns:
            print("Kurwa, żadna z podanych kolumn nie istnieje w danych!")
            return self.data
        
        # Wybierz metodę skalowania
        if method == 'standard':
            scaler = StandardScaler()
            method_name = "standaryzację"
        elif method == 'minmax':
            scaler = MinMaxScaler()
            method_name = "normalizację do przedziału [0,1]"
        else:
            print(f"Kurwa, nieznana metoda skalowania: {method}")
            return self.data
        
        # Wykonaj skalowanie
        self.data[existing_columns] = scaler.fit_transform(self.data[existing_columns])
        
        self.transformations.append(f"Wykonano {method_name} dla kolumn: {existing_columns}")
        print(f"Kurwa, wykonano {method_name} dla kolumn: {existing_columns}")
        
        return self.data
    
    def encode_categorical(self, columns=None, method='onehot'):
        """
        Koduje cechy kategoryczne.
        
        Args:
            columns (list, optional): Lista kolumn do kodowania. Jeśli None, używa wszystkich kategorycznych.
            method (str): Metoda kodowania ('onehot', 'label')
        
        Returns:
            pandas.DataFrame: Dane z zakodowanymi cechami kategorycznymi
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.categorical_columns
        
        if not columns:
            print("Kurwa, brak kolumn kategorycznych do kodowania!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if not existing_columns:
            print("Kurwa, żadna z podanych kolumn nie istnieje w danych!")
            return self.data
        
        if method == 'onehot':
            # One-hot encoding
            for col in existing_columns:
                # Zastosuj one-hot encoding
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=False)
                # Dodaj zakodowane kolumny do danych
                self.data = pd.concat([self.data, dummies], axis=1)
                # Usuń oryginalną kolumnę
                self.data = self.data.drop(col, axis=1)
            
            self.transformations.append(f"Wykonano kodowanie one-hot dla kolumn: {existing_columns}")
            print(f"Kurwa, wykonano kodowanie one-hot dla kolumn: {existing_columns}")
            
        elif method == 'label':
            # Label encoding (zastępuje wartości kategoryczne liczbami)
            for col in existing_columns:
                self.data[col] = self.data[col].astype('category').cat.codes
            
            self.transformations.append(f"Wykonano kodowanie etykiet dla kolumn: {existing_columns}")
            print(f"Kurwa, wykonano kodowanie etykiet dla kolumn: {existing_columns}")
            
        else:
            print(f"Kurwa, nieznana metoda kodowania: {method}")
        
        # Zaktualizuj listy kolumn
        self._identify_column_types()
        
        return self.data
    
    def remove_outliers(self, columns=None, method='iqr', threshold=1.5):
        """
        Usuwa wartości odstające z danych.
        
        Args:
            columns (list, optional): Lista kolumn do analizy. Jeśli None, używa wszystkich numerycznych.
            method (str): Metoda wykrywania ('iqr', 'zscore')
            threshold (float): Próg dla IQR (typowo 1.5) lub Z-score (typowo 3)
        
        Returns:
            pandas.DataFrame: Dane bez wartości odstających
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("Kurwa, brak kolumn numerycznych do usunięcia wartości odstających!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if not existing_columns:
            print("Kurwa, żadna z podanych kolumn nie istnieje w danych!")
            return self.data
        
        initial_rows = len(self.data)
        outliers_indices = set()
        
        for col in existing_columns:
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
                
                # Znajdź indeksy wartości odstających
                outliers = self.data[
                    (self.data[col] < lower_bound) | 
                    (self.data[col] > upper_bound)
                ].index.tolist()
                
                outliers_indices.update(outliers)
                
            elif method == 'zscore':
                # Metoda Z-score
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = self.data[z_scores > threshold].index.tolist()
                outliers_indices.update(outliers)
        
        # Usuń wszystkie wiersze z wartościami odstającymi
        self.data = self.data.drop(list(outliers_indices))
        
        removed_rows = initial_rows - len(self.data)
        self.transformations.append(f"Usunięto {removed_rows} wierszy z wartościami odstającymi (metoda {method})")
        print(f"Kurwa, usunięto {removed_rows} wierszy z wartościami odstającymi (metoda {method})")
        
        return self.data
    
    def select_features(self, target_column, k=5, method='f_regression'):
        """
        Wybiera najważniejsze cechy za pomocą metod statystycznych.
        
        Args:
            target_column (str): Nazwa kolumny docelowej
            k (int): Liczba cech do wybrania
            method (str): Metoda selekcji ('f_regression', 'mutual_info')
        
        Returns:
            pandas.DataFrame: Dane z wybranymi cechami
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if target_column not in self.data.columns:
            print(f"Kurwa, kolumna docelowa {target_column} nie istnieje w danych!")
            return self.data
        
        # Wybierz dostępne kolumny numeryczne (bez kolumny docelowej)
        features = [col for col in self.numeric_columns if col != target_column]
        
        if not features:
            print("Kurwa, brak cech numerycznych do selekcji!")
            return self.data
        
        # Przygotuj dane
        X = self.data[features]
        y = self.data[target_column]
        
        # Wybierz metodę selekcji
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(features)))
            method_name = "F-regression"
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(features)))
            method_name = "Mutual Information"
        else:
            print(f"Kurwa, nieznana metoda selekcji cech: {method}")
            return self.data
        
        # Wykonaj selekcję
        X_new = selector.fit_transform(X, y)
        
        # Pobierz nazwy wybranych cech
        selected_indices = selector.get_support(indices=True)
        selected_features = [features[i] for i in selected_indices]
        
        # Wyświetl wyniki selekcji
        print(f"Kurwa, wybrano {len(selected_features)} najważniejszych cech:")
        for i, feature in enumerate(selected_features):
            score = selector.scores_[selected_indices[i]]
            print(f"  {feature}: {score:.4f}")
        
        # Utwórz nowy dataframe z wybranymi cechami i kolumną docelową
        selected_features.append(target_column)
        self.data = self.data[selected_features]
        
        self.transformations.append(f"Wybrano {len(selected_features)-1} najważniejszych cech metodą {method_name}")
        
        return self.data
    
    def create_polynomial_features(self, columns=None, degree=2):
        """
        Tworzy cechy wielomianowe.
        
        Args:
            columns (list, optional): Lista kolumn do transformacji. Jeśli None, używa wszystkich numerycznych.
            degree (int): Stopień wielomianu
        
        Returns:
            pandas.DataFrame: Dane z dodanymi cechami wielomianowymi
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("Kurwa, brak kolumn numerycznych do transformacji!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if not existing_columns:
            print("Kurwa, żadna z podanych kolumn nie istnieje w danych!")
            return self.data
        
        # Twórz cechy wielomianowe
        for col in existing_columns:
            for d in range(2, degree + 1):
                new_col_name = f"{col}^{d}"
                self.data[new_col_name] = self.data[col] ** d
        
        self.transformations.append(f"Utworzono cechy wielomianowe (stopnia {degree}) dla kolumn: {existing_columns}")
        print(f"Kurwa, utworzono cechy wielomianowe (stopnia {degree}) dla kolumn: {existing_columns}")
        
        # Zaktualizuj listy kolumn
        self._identify_column_types()
        
        return self.data
    
    def create_interaction_features(self, columns=None):
        """
        Tworzy cechy interakcyjne (iloczyny par cech).
        
        Args:
            columns (list, optional): Lista kolumn do interakcji. Jeśli None, używa wszystkich numerycznych.
        
        Returns:
            pandas.DataFrame: Dane z dodanymi cechami interakcyjnymi
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns or len(columns) < 2:
            print("Kurwa, potrzeba przynajmniej 2 kolumn numerycznych do interakcji!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if len(existing_columns) < 2:
            print("Kurwa, potrzeba przynajmniej 2 istniejących kolumn do interakcji!")
            return self.data
        
        # Twórz cechy interakcyjne
        interaction_count = 0
        for i in range(len(existing_columns)):
            for j in range(i+1, len(existing_columns)):
                col1, col2 = existing_columns[i], existing_columns[j]
                interaction_name = f"{col1}*{col2}"
                self.data[interaction_name] = self.data[col1] * self.data[col2]
                interaction_count += 1
        
        self.transformations.append(f"Utworzono {interaction_count} cech interakcyjnych")
        print(f"Kurwa, utworzono {interaction_count} cech interakcyjnych")
        
        # Zaktualizuj listy kolumn
        self._identify_column_types()
        
        return self.data
    
    def bin_numeric_features(self, columns=None, bins=5, strategy='uniform'):
        """
        Dyskretyzuje (dzieli na przedziały) cechy numeryczne.
        
        Args:
            columns (list, optional): Lista kolumn do dyskretyzacji. Jeśli None, używa wszystkich numerycznych.
            bins (int): Liczba przedziałów
            strategy (str): Strategia podziału ('uniform', 'quantile')
        
        Returns:
            pandas.DataFrame: Dane z dyskretyzowanymi cechami
        """
        if self.data is None:
            print("Kurwa, brak danych do przetworzenia!")
            return None
        
        if columns is None:
            columns = self.numeric_columns
        
        if not columns:
            print("Kurwa, brak kolumn numerycznych do dyskretyzacji!")
            return self.data
        
        # Sprawdź, czy kolumny istnieją w danych
        existing_columns = [col for col in columns if col in self.data.columns]
        if not existing_columns:
            print("Kurwa, żadna z podanych kolumn nie istnieje w danych!")
            return self.data
        
        for col in existing_columns:
            if strategy == 'uniform':
                # Podział równomierny
                bin_edges = np.linspace(
                    self.data[col].min(), 
                    self.data[col].max(), 
                    bins + 1
                )
            elif strategy == 'quantile':
                # Podział kwantylowy
                bin_edges = pd.qcut(
                    self.data[col], 
                    q=bins, 
                    retbins=True, 
                    duplicates='drop'
                )[1]
            else:
                print(f"Kurwa, nieznana strategia dyskretyzacji: {strategy}")
                continue
            
            # Dodaj nową kolumnę z dyskretyzowanymi wartościami
            binned_col_name = f"{col}_binned"
            self.data[binned_col_name] = pd.cut(
                self.data[col], 
                bins=bin_edges, 
                labels=range(len(bin_edges)-1), 
                include_lowest=True
            )
        
        self.transformations.append(f"Dyskretyzowano kolumny {existing_columns} na {bins} przedziałów (strategia: {strategy})")
        print(f"Kurwa, dyskretyzowano kolumny {existing_columns} na {bins} przedziałów (strategia: {strategy})")
        
        # Zaktualizuj listy kolumn
        self._identify_column_types()
        
        return self.data
    
    def get_transformation_log(self):
        """
        Zwraca log transformacji wykonanych na danych.
        
        Returns:
            list: Lista wykonanych transformacji
        """
        return self.transformations
    
    def reset_data(self):
        """
        Przywraca oryginalne dane.
        
        Returns:
            pandas.DataFrame: Oryginalne dane
        """
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.transformations = []
            self._identify_column_types()
            print("Kurwa, przywrócono oryginalne dane")
            return self.data
        else:
            print("Kurwa, brak oryginalnych danych do przywrócenia!")
            return None
    
    def save_data(self, file_path):
        """
        Zapisuje przetworzone dane do pliku CSV.
        
        Args:
            file_path (str): Ścieżka do pliku wyjściowego
        
        Returns:
            bool: True jeśli zapis się powiódł, False w przeciwnym przypadku
        """
        if self.data is None:
            print("Kurwa, brak danych do zapisania!")
            return False
        
        try:
            # Utwórz katalogi, jeśli nie istnieją
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Zapisz dane
            self.data.to_csv(file_path, index=False)
            print(f"Kurwa, zapisano przetworzone dane do {file_path}")
            return True
        except Exception as e:
            print(f"Kurwa, błąd podczas zapisywania danych: {e}")
            return False


def main():
    """Przykładowe użycie procesora danych."""
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
        
        # Dodaj kilka wartości odstających
        df.at[0, 'x1'] = 10.0  # Wartość odstająca
        df.at[1, 'x2'] = 15.0  # Wartość odstająca
        
        # Dodaj kilka brakujących wartości
        df.at[2, 'x1'] = np.nan
        df.at[3, 'x2'] = np.nan
        df.at[4, 'category'] = None
        
        # Dodaj duplikaty
        df = pd.concat([df, df.iloc[5:10].copy()])
        
        df.to_csv(data_path, index=False)
        print(f"Kurwa, zapisano przykładowe dane do {data_path}")
    
    # Inicjalizacja procesora
    processor = DataProcessor()
    
    # Wczytanie danych
    data = processor.load_data(data_path)
    
    if data is not None:
        # Przykładowe przetwarzanie
        processor.remove_duplicates()
        processor.handle_missing_values()
        processor.remove_outliers(method='iqr')
        processor.scale_features(method='standard')
        processor.encode_categorical(method='onehot')
        processor.create_polynomial_features(degree=2)
        processor.create_interaction_features()
        
        # Pokaż log transformacji
        print("\nKurwa, wykonane transformacje:")
        for i, transform in enumerate(processor.get_transformation_log(), 1):
            print(f"{i}. {transform}")
        
        # Zapisz przetworzone dane
        processor.save_data("dane/przetworzone.csv")


if __name__ == "__main__":
    main() 