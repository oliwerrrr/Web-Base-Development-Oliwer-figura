# Podsumowanie analizy ruchu internetowego

## Główne wyniki analizy

### Prędkości pobierania i wysyłania danych

- **Średnia prędkość pobierania (download)**: 31 001 327,37 bajtów/s (~31 MB/s)
- **Średnia prędkość wysyłania (upload)**: 9 777 732,40 bajtów/s (~9,8 MB/s)
- **Mediana prędkości pobierania**: 14 366 030,00 bajtów/s (~14,4 MB/s)
- **Mediana prędkości wysyłania**: 2 168 848,00 bajtów/s (~2,2 MB/s)
- **Stosunek średnich (DL/UL)**: 3,17 (pobieranie jest średnio 3,17 razy szybsze)
- **Stosunek median (DL/UL)**: 6,62 (mediana pobierania jest 6,62 razy wyższa)

### Korelacje i czynniki wpływające na wydajność

- **Korelacja między opóźnieniem a prędkością pobierania**: -0,3354 (ujemna korelacja)
- Każdy dodatkowy 1ms opóźnienia zmniejsza prędkość pobierania o około 27,44 bajtów/s
- Model regresji wskazuje, że opóźnienie wyjaśnia około 11,42% zmienności w prędkości pobierania (R² = 0,1142)

## Analiza danych wejściowych

Przeanalizowano różne zestawy danych ruchu internetowego:

- **Dane pobierania**: 
  - httpgetmt: 724 511 pomiarów
  - httpgetmt6: 489 pomiarów
  - dlping: 1 246 832 pomiarów
  - webget: 9 522 842 pomiarów

- **Dane wysyłania**:
  - httppostmt: 722 118 pomiarów
  - httppostmt6: 1 159 pomiarów
  - ulping: 1 260 944 pomiarów

- **Dodatkowe dane sieci** (analizowane częściowo, po 50 000 pomiarów):
  - dns, ping, traceroute, udplatency, udpjitter, udpcloss

## Kluczowe wnioski

1. **Asymetria prędkości** - występuje znacząca asymetria między prędkościami pobierania i wysyłania, co jest typowe dla większości łączy internetowych.

2. **Wpływ opóźnienia** - potwierdzono, że większe opóźnienie (latencja) wpływa negatywnie na prędkość pobierania danych.

3. **Rozkład prędkości** - rozkłady prędkości zarówno pobierania, jak i wysyłania są prawoskośne, co oznacza, że większość użytkowników doświadcza prędkości poniżej średniej.

4. **Zmienność w czasie** - analiza czasowa pokazuje, że prędkości pobierania i wysyłania mogą się znacząco zmieniać w różnych okresach.

## Wygenerowane wizualizacje

Analiza wygenerowała szereg wizualizacji, które pomagają lepiej zrozumieć dane:

1. Histogramy rozkładu prędkości pobierania i wysyłania
2. Wykresy czasowe zmian prędkości
3. Wykresy pudełkowe porównujące rozkłady prędkości
4. Wykres pokazujący korelację między opóźnieniem a prędkością pobierania

## Zalecenia

1. **Optymalizacja asymetrii** - rozważenie konfiguracji oferujących bardziej zrównoważone prędkości pobierania i wysyłania dla użytkowników, którzy często wysyłają duże ilości danych.

2. **Minimalizacja opóźnień** - podjęcie działań mających na celu redukcję opóźnień w sieci, co przełoży się na poprawę prędkości pobierania.

3. **Regularne monitorowanie** - zaleca się systematyczne przeprowadzanie podobnych analiz w celu śledzenia zmian w wydajności sieci w czasie.

4. **Głębsza analiza korelacji** - warto rozszerzyć analizę o badanie wpływu innych parametrów sieci (np. jitter, packet loss) na prędkości transmisji danych. 