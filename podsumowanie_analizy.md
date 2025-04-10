# Podsumowanie analizy ruchu internetowego 2021 vs 2023

## Wykonane kroki

1. **Analiza danych z 2021 roku**
   - Wczytanie danych z katalogu `data_2021`
   - Przeprowadzenie analizy ruchu internetowego
   - Generowanie wykresów i raportów w katalogu `wyniki_ruchu_internetowego_2021`

2. **Analiza danych z 2023 roku**
   - Wczytanie danych z katalogu `data`
   - Przeprowadzenie analizy ruchu internetowego
   - Generowanie wykresów i raportów w katalogu `wyniki_ruchu_internetowego_2023`

3. **Aktualizacja wizualizacji**
   - Modyfikacja skryptu `update_visualizations.py` do obsługi danych z 2021 roku
   - Generowanie spójnych wykresów z angielskimi opisami dla obu zestawów danych

4. **Porównanie wyników**
   - Utworzenie skryptu `compare_2021_2023.py` do porównania wyników
   - Generowanie wykresów porównawczych
   - Utworzenie raportu podsumowującego zmiany między 2021 a 2023 rokiem

## Główne wyniki porównawcze

### Prędkość pobierania (Download)
- 2021: 12.36 MB/s
- 2023: 31.00 MB/s
- **Zmiana: +150.77%**

### Prędkość wysyłania (Upload)
- 2021: 2.94 MB/s
- 2023: 9.78 MB/s
- **Zmiana: +232.81%**

### Stosunek prędkości pobierania do wysyłania
- 2021: 4.21 (większa dysproporcja)
- 2023: 3.17 (mniejsza dysproporcja)
- **Zmiana: -24.70%**

### Wpływ opóźnienia na prędkość pobierania
- 2021: korelacja -0.1794 (słabszy wpływ)
- 2023: korelacja -0.3354 (silniejszy wpływ)
- **Zmiana w sile korelacji: +86.96%**

## Wnioski

1. **Znacząca poprawa wydajności internetu**
   - Zarówno prędkość pobierania, jak i wysyłania uległy drastycznej poprawie (ponad dwukrotnie)
   - Prędkość wysyłania poprawiła się w większym stopniu niż prędkość pobierania

2. **Bardziej symetryczne łącze**
   - Stosunek prędkości pobierania do wysyłania zmniejszył się o 24.7%
   - Wskazuje to na bardziej zrównoważone łącze internetowe w 2023 roku

3. **Silniejszy wpływ opóźnienia sieci**
   - W 2023 roku opóźnienie sieci ma większy negatywny wpływ na prędkość pobierania
   - Siła korelacji wzrosła o prawie 87%
   - Sugeruje to większą wrażliwość nowszych protokołów/aplikacji na opóźnienia

## Pliki wynikowe

1. **Katalogi z wynikami analiz**
   - `wyniki_ruchu_internetowego_2021` - wyniki dla danych z 2021 roku
   - `wyniki_ruchu_internetowego_2023` - wyniki dla danych z 2023 roku
   - `comparison_2021_2023` - wykresy i raport porównawczy

2. **Kluczowe pliki porównawcze**
   - `comparison_2021_2023/comparison_report.md` - raport tekstowy z porównaniem
   - `comparison_2021_2023/speed_comparison.png` - wykres porównujący prędkości
   - `comparison_2021_2023/speed_ratio_comparison.png` - wykres porównujący proporcje prędkości
   - `comparison_2021_2023/latency_correlation_comparison.png` - wykres porównujący wpływ opóźnienia

## Podsumowanie

Analiza wykazała znaczącą poprawę wydajności internetowej między rokiem 2021 a 2023. Łącza internetowe stały się nie tylko szybsze, ale również bardziej symetryczne. Jednocześnie wzrósł wpływ opóźnień sieciowych na faktyczną prędkość, co może wskazywać na rosnącą rolę jakości połączenia, a nie tylko jego teoretycznej przepustowości. 