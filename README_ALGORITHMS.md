# Dokumentacja Algorytmów Analizy Danych

## Spis treści

1. [Wprowadzenie](#wprowadzenie)
2. [Analiza Danych (DA)](#analiza-danych-da)
3. [Eksploracja Danych (DE)](#eksploracja-danych-de)
4. [Eksploracyjna Analiza Danych (EDA)](#eksploracyjna-analiza-danych-eda)
5. [Korelacja Statystyczna](#korelacja-statystyczna)
6. [Regresja Liniowa](#regresja-liniowa)
7. [Parametry Ilościowe i Jakościowe](#parametry-ilościowe-i-jakościowe)
8. [Wartości Odstające](#wartości-odstające)
9. [Kwartyle i Rozstęp Międzykwartylowy (IQR)](#kwartyle-i-rozstęp-międzykwartylowy-iqr)
10. [Mapy Ciepła](#mapy-ciepła)
11. [Przetwarzanie Danych](#przetwarzanie-danych)
12. [Przykłady Użycia](#przykłady-użycia)

## Wprowadzenie

Ten dokument opisuje algorytmy i techniki analizy danych zaimplementowane w projekcie. Biblioteka zawiera dwa główne moduły:

- `data_analysis.py` - zawiera klasę `DataAnalyzer` do analizy i wizualizacji danych
- `data_processor.py` - zawiera klasę `DataProcessor` do przetwarzania i transformacji danych

## Analiza Danych (DA)

Analiza danych (Data Analysis) to proces badania, czyszczenia, przekształcania i modelowania danych w celu odkrycia użytecznych informacji, wyciągnięcia wniosków i wsparcia podejmowania decyzji.

### Zaimplementowane funkcje:

- Wczytywanie danych z plików CSV
- Podstawowe statystyki opisowe
- Wizualizacja danych
- Analiza korelacji między zmiennymi
- Modelowanie regresji liniowej

## Eksploracja Danych (DE)

Eksploracja danych (Data Exploration) to pierwszy krok w analizie danych, polegający na zapoznaniu się z danymi, ich strukturą i podstawowymi właściwościami.

### Zaimplementowane funkcje:

- Zbadanie struktury danych (wymiary, kolumny, typy danych)
- Wykrywanie brakujących wartości
- Podstawowe statystyki dla kolumn numerycznych i kategorycznych
- Generowanie raportów tekstowych z wynikami eksploracji

## Eksploracyjna Analiza Danych (EDA)

Eksploracyjna analiza danych (Exploratory Data Analysis) to podejście do analizy zestawów danych, które wykorzystuje techniki wizualizacji do odkrywania wzorców, identyfikacji anomalii i testowania hipotez.

### Zaimplementowane funkcje:

- Histogramy dla zmiennych numerycznych
- Wykresy pudełkowe (boxplots) dla wykrywania wartości odstających
- Wykresy rozproszenia (scatterplots) dla par zmiennych
- Mapy ciepła korelacji

## Korelacja Statystyczna

Korelacja statystyczna to miara zależności między dwoma zmiennymi. Najczęściej używane metody korelacji to korelacja Pearsona, Spearmana i Kendalla.

### Zaimplementowane funkcje:

- Obliczanie macierzy korelacji Pearsona
- Obliczanie macierzy korelacji Spearmana
- Obliczanie macierzy korelacji Kendalla
- Wizualizacja macierzy korelacji jako map ciepła

## Regresja Liniowa

Regresja liniowa to technika modelowania statystycznego używana do przewidywania wartości zmiennej zależnej na podstawie jednej lub więcej zmiennych niezależnych.

### Zaimplementowane funkcje:

- Budowa modelu regresji liniowej
- Podział danych na zbiór treningowy i testowy
- Ocena modelu przy użyciu różnych metryk:
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - R² (współczynnik determinacji)
- Wizualizacja linii regresji i danych

## Parametry Ilościowe i Jakościowe

Dane można podzielić na parametry ilościowe (numeryczne) i jakościowe (kategoryczne).

### Obsługa parametrów ilościowych:

- Statystyki opisowe: średnia, mediana, odchylenie standardowe, itp.
- Skalowanie danych (standaryzacja, normalizacja)
- Tworzenie cech wielomianowych i interakcyjnych

### Obsługa parametrów jakościowych:

- Kodowanie one-hot
- Kodowanie etykiet
- Statystyki dla danych kategorycznych (liczebność, częstość)

## Wartości Odstające

Wartości odstające (outliers) to obserwacje, które znacznie różnią się od innych obserwacji w danych.

### Zaimplementowane metody wykrywania:

- Metoda IQR (Interquartile Range)
- Metoda Z-score
- Wizualizacja wartości odstających na wykresach pudełkowych

## Kwartyle i Rozstęp Międzykwartylowy (IQR)

Kwartyle dzielą zbiór danych na cztery równe części. Rozstęp międzykwartylowy (IQR) to różnica między trzecim (Q3) a pierwszym (Q1) kwartylem.

### Zaimplementowane funkcje:

- Obliczanie kwartyli (Q1, Q2/mediana, Q3)
- Obliczanie rozstępu międzykwartylowego (IQR)
- Analiza kwartylowa dla wszystkich zmiennych numerycznych
- Wizualizacja kwartyli na wykresach pudełkowych

## Mapy Ciepła

Mapy ciepła (heatmaps) to graficzne reprezentacje danych, gdzie wartości są reprezentowane przez kolory. Są szczególnie przydatne do wizualizacji macierzy korelacji.

### Zaimplementowane funkcje:

- Tworzenie map ciepła dla macierzy danych
- Dostosowywanie kolorów i etykiet
- Wizualizacja macierzy korelacji jako map ciepła

## Przetwarzanie Danych

Moduł `data_processor.py` zawiera funkcje do przetwarzania i czyszczenia danych przed analizą.

### Zaimplementowane funkcje:

- Usuwanie duplikatów
- Obsługa brakujących wartości
- Skalowanie cech numerycznych
- Kodowanie cech kategorycznych
- Usuwanie wartości odstających
- Selekcja cech
- Tworzenie cech wielomianowych i interakcyjnych
- Dyskretyzacja zmiennych numerycznych

## Przykłady Użycia

### Analiza danych

```python
from data_analysis import DataAnalyzer

# Inicjalizacja analizatora
analyzer = DataAnalyzer(data_path="dane/przyklad.csv")

# Wczytanie danych
df = analyzer.load_data()

# Eksploracja danych
analyzer.data_exploration()

# Wizualizacje EDA
analyzer.eda_visualizations()

# Analiza korelacji
correlation_matrix = analyzer.correlation_analysis(method='pearson')

# Regresja liniowa
regression_results = analyzer.linear_regression('x1', 'y')

# Analiza wartości odstających
outliers_info = analyzer.detect_outliers(method='iqr')

# Analiza kwartylowa
quartile_info = analyzer.quartile_analysis()
```

### Przetwarzanie danych

```python
from data_processor import DataProcessor

# Inicjalizacja procesora
processor = DataProcessor()

# Wczytanie danych
data = processor.load_data("dane/przyklad.csv")

# Przetwarzanie danych
processor.remove_duplicates()
processor.handle_missing_values(strategy='mean')
processor.remove_outliers(method='iqr')
processor.scale_features(method='standard')
processor.encode_categorical(method='onehot')
processor.create_polynomial_features(degree=2)
processor.create_interaction_features()

# Zapisanie przetworzonych danych
processor.save_data("dane/przetworzone.csv")
```

---

Dokumentacja przygotowana dla projektu Web Base Development. 