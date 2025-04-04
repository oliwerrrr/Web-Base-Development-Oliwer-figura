# Zajebista dokumentacja algorytmów i wizualizacji

Ten dokument zawiera szczegółowy opis wszystkich algorytmów i typów wizualizacji używanych w naszym projekcie. Kurwa, to będzie dobre!

## Spis treści

1. [Rodzaje wizualizacji](#rodzaje-wizualizacji)
   - [Histogramy](#histogramy)
   - [Wykresy funkcji](#wykresy-funkcji)
   - [Mapy ciepła](#mapy-ciepła)
2. [Algorytmy i ich wizualizacje](#algorytmy-i-ich-wizualizacje)
   - [Rozkłady statystyczne](#rozkłady-statystyczne)
   - [Funkcje matematyczne](#funkcje-matematyczne)
   - [Algorytmy generujące mapy ciepła](#algorytmy-generujące-mapy-ciepła)
3. [Interpretacja wizualizacji](#interpretacja-wizualizacji)
4. [Tworzenie własnych wizualizacji](#tworzenie-własnych-wizualizacji)

## Rodzaje wizualizacji

### Histogramy

Histogram to, kurwa, zajebisty sposób na pokazanie rozkładu wartości w zbiorze danych. Składa się z:
- **Oś X**: Kategorie lub przedziały wartości
- **Oś Y**: Częstotliwość lub liczba wystąpień
- **Słupki**: Reprezentują liczbę elementów w każdej kategorii

#### Przykłady w projekcie:

- **histogram_normalny.png** - Pokazuje rozkład normalny (Gaussa) dla losowo generowanych wartości
- **histogram_poisson.png** - Przedstawia rozkład Poissona, który modeluje rzadkie, losowe zdarzenia

### Wykresy funkcji

Wykresy funkcji to graficzna reprezentacja zależności między dwiema zmiennymi. Składają się z:
- **Oś X**: Wartości wejściowe (dziedzina funkcji)
- **Oś Y**: Wartości wyjściowe (zbiór wartości funkcji)
- **Punkty/linia**: Pokazują zależność między x i y

#### Przykłady w projekcie:

- **wykres_sin.png** - Wykres funkcji sinus w przedziale [0, 10]
- **wykres_kwadratowy.png** - Wykres funkcji kwadratowej x² / 10 w przedziale [0, 10]

### Mapy ciepła

Mapy ciepła to, kurwa, najlepszy sposób na wizualizację danych dwuwymiarowych, gdzie wartość każdego punktu (i,j) oznaczona jest kolorem. Składają się z:
- **Siatka komórek**: Reprezentuje punkty danych
- **Kolorystyka**: Intensywność koloru odpowiada wartościom danych
- **Legenda kolorów**: Określa mapowanie wartości na kolory

#### Przykłady w projekcie:

- **mapa_ciepla_losowa.png** - Mapa ciepła z losowymi wartościami
- **mapa_ciepla_sincos.png** - Mapa ciepła generowana przez funkcję sin(i/5) * cos(j/5)

## Algorytmy i ich wizualizacje

### Rozkłady statystyczne

#### Rozkład normalny (Gaussa)

**Co to, kurwa, jest?**
Rozkład normalny to jeden z najważniejszych rozkładów prawdopodobieństwa w statystyce. Opisuje zjawiska, w których większość wartości skupia się wokół średniej.

**Jak to działa:**
- Generowanie próbek: `np.random.normal(średnia, odchylenie_standardowe, liczba_próbek)`
- W naszym przypadku: `np.random.normal(50, 10, 10)` - średnia 50, odchylenie 10, 10 próbek

**Zastosowania:**
- Modelowanie błędów pomiarowych
- Szacowanie niepewności
- Podstawa dla testów statystycznych (t-test, z-test)
- Modelowanie zjawisk naturalnych

**Interpretacja wizualizacji:**
- Charakterystyczny kształt "dzwonu"
- Symetryczny rozkład wokół średniej
- Większość wartości w okolicy średniej

#### Rozkład Poissona

**Co to, kurwa, jest?**
Rozkład Poissona modeluje liczbę zdarzeń występujących w ustalonym przedziale czasu lub przestrzeni, przy założeniu, że zdarzenia te zachodzą ze stałą średnią częstością i niezależnie od siebie.

**Jak to działa:**
- Generowanie próbek: `np.random.poisson(lambda, liczba_próbek)`
- W naszym przypadku: `np.random.poisson(5, 10)` - średnia 5, 10 próbek

**Zastosowania:**
- Modelowanie liczby przychodzących połączeń telefonicznych
- Analiza liczby wypadków lub katastrof
- Modelowanie rzadkich zdarzeń (liczba mutacji DNA, liczba awarii)
- Procesy kolejkowe

**Interpretacja wizualizacji:**
- Dyskretny rozkład (tylko liczby całkowite)
- Zskośny (dla małych wartości lambda)
- Dla dużych lambda zbliża się do rozkładu normalnego

### Funkcje matematyczne

#### Funkcja sinus

**Co to, kurwa, jest?**
Podstawowa funkcja trygonometryczna opisująca falę harmoniczną.

**Definicja matematyczna:**
- y = sin(x)
- W naszym przypadku: `np.sin(x_data1)` gdzie `x_data1 = np.linspace(0, 10, 20)`

**Zastosowania:**
- Modelowanie drgań i fal
- Analiza sygnałów
- Przewidywanie zjawisk cyklicznych
- Przetwarzanie dźwięku i obrazu

**Interpretacja wizualizacji:**
- Okresowa funkcja z okresem 2π
- Wartości w zakresie [-1, 1]
- Symetryczna względem osi Y

#### Funkcja kwadratowa

**Co to, kurwa, jest?**
Funkcja drugiego stopnia postaci y = ax² + bx + c.

**Definicja matematyczna:**
- W naszym przypadku: `y_data2 = x_data2 ** 2 / 10` gdzie `x_data2 = np.linspace(0, 10, 20)`
- Uproszczona postać: y = x²/10

**Zastosowania:**
- Modelowanie ruchu w polu grawitacyjnym
- Analiza kosztów i zysków
- Optymalizacja
- Modelowanie procesów z przyspieszeniem

**Interpretacja wizualizacji:**
- Paraboliczny kształt
- Minimum w punkcie (0,0)
- Monotoniczna dla x > 0
- Wartość rośnie coraz szybciej wraz ze wzrostem x

### Algorytmy generujące mapy ciepła

#### Losowa mapa ciepła

**Co to, kurwa, jest?**
Mapa ciepła generowana przez wartości losowe z rozkładu jednostajnego.

**Jak to działa:**
- Generowanie danych: `np.random.rand(10, 10)` - macierz 10x10 z losowymi wartościami [0,1]
- Kolorowanie: Wartości mapowane na kolory (w naszym przypadku "hot" - odcienie czerwieni i żółci)

**Zastosowania:**
- Testowanie algorytmów wizualizacji
- Modelowanie szumu
- Generowanie losowych terranów/map
- Symulacje losowych procesów

**Interpretacja wizualizacji:**
- Brak widocznego wzorca (szum)
- Losowy rozkład kolorów
- Różnice w intensywności odpowiadają losowym wartościom

#### Mapa ciepła sin-cos

**Co to, kurwa, jest?**
Mapa ciepła generowana przez funkcję matematyczną sin(i/5) * cos(j/5).

**Jak to działa:**
```python
for i in range(10):
    for j in range(10):
        heatmap_data2[i, j] = math.sin(i/5) * math.cos(j/5)
```

**Zastosowania:**
- Wizualizacja pól wektorowych
- Modelowanie interferencji fal
- Wizualizacja funkcji dwóch zmiennych
- Analiza falna i spektralna

**Interpretacja wizualizacji:**
- Okresowy wzór z widocznymi maksimami i minimami
- Struktura "szachownicy" z przeciwnymi wartościami
- Płynne przejścia kolorów dzięki ciągłości funkcji

## Interpretacja wizualizacji

### Jak czytać histogramy:
1. **Kurwa, najpierw spójrz na oś X** - Co reprezentują poszczególne słupki?
2. **Potem na oś Y** - Ile elementów/zdarzeń zawiera każdy słupek?
3. **Znajdź najwyższy słupek** - To najczęstsza wartość (modalna)
4. **Ocen kształt** - Czy rozkład jest symetryczny? Skośny? Wielomodalny?
5. **Zwróć uwagę na wartości odstające** - Czy są słupki znacznie wyższe/niższe od pozostałych?

### Jak czytać wykresy funkcji:
1. **Kurwa, najpierw zidentyfikuj funkcję** - Jaki jest kształt krzywej?
2. **Znajdź ekstrema** - Gdzie funkcja osiąga maksima i minima?
3. **Oceń monotoniczność** - Gdzie funkcja rośnie, a gdzie maleje?
4. **Zwróć uwagę na asymptoty i osobliwości** - Czy funkcja dąży do nieskończoności?
5. **Sprawdź przecięcia z osiami** - Gdzie funkcja przyjmuje wartość zero?

### Jak czytać mapy ciepła:
1. **Kurwa, najpierw sprawdź skalę kolorów** - Co oznaczają poszczególne kolory?
2. **Znajdź ekstrema** - Gdzie występują najmniejsze i największe wartości?
3. **Szukaj wzorów** - Czy widać powtarzające się struktury?
4. **Analizuj gradienty** - Gdzie wartości zmieniają się najszybciej?
5. **Interpretuj w kontekście danych** - Co oznaczają obserwowane wzorce?

## Tworzenie własnych wizualizacji

Jeśli chcesz, kurwa, dodać własne wizualizacje do projektu, postępuj zgodnie z tymi krokami:

### Dodawanie nowego histogramu:
```python
from PIL import Image, ImageDraw
import numpy as np

# Generuj dane
data = np.random.YOUR_DISTRIBUTION(params)

# Twórz histogram
img = Image.new('RGB', (500, 400), color='white')
draw = ImageDraw.Draw(img)

# Rysuj słupki
max_value = max(data)
bar_width = 30
gap = 10

for i, value in enumerate(data):
    bar_height = int((value / max_value) * 300)
    x0 = 50 + i * (bar_width + gap)
    y0 = 350 - bar_height
    x1 = x0 + bar_width
    y1 = 350
    draw.rectangle([x0, y0, x1, y1], fill="color", outline='black')

# Zapisz plik
img.save("sciezka/do/twojego_histogramu.png")
```

### Dodawanie nowego wykresu funkcji:
```python
from PIL import Image, ImageDraw
import numpy as np

# Generuj dane
x_values = np.linspace(start, stop, num_points)
y_values = YOUR_FUNCTION(x_values)

# Twórz wykres
img = Image.new('RGB', (500, 400), color='white')
draw = ImageDraw.Draw(img)

# Skaluj dane do wymiarów obrazu
# [kod skalowania]

# Rysuj punkty i linie
# [kod rysowania]

# Zapisz plik
img.save("sciezka/do/twojego_wykresu.png")
```

### Dodawanie nowej mapy ciepła:
```python
from PIL import Image, ImageDraw
import numpy as np

# Generuj dane 2D
data = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        data[i, j] = YOUR_FUNCTION(i, j)

# Twórz mapę ciepła
# [kod tworzenia mapy ciepła]

# Zapisz plik
img.save("sciezka/do/twojej_mapy_ciepla.png")
```

## Podsumowanie

Kurwa, teraz masz pełen obraz wszystkich wizualizacji i algorytmów w projekcie! Możesz:
1. Przeglądać istniejące wizualizacje
2. Interpretować ich znaczenie
3. Tworzyć własne wizualizacje
4. Wykorzystywać różne algorytmy i rozkłady statystyczne

Jeśli masz jakieś pytania, pamiętaj że zawsze możesz:
- Przejrzeć kod źródłowy w `test_visualization_viewer.py`
- Uruchomić przeglądarkę wizualizacji: `python run.py`
- Wygenerować nowe przykłady: `python run.py --gen-samples` 