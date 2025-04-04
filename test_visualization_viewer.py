#!/usr/bin/env python3
import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from PyQt5.QtWidgets import QApplication, QPushButton
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt, QTimer

# Importuj przegladarkę wizualizacji
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualization_viewer_qt import VisualizationViewer


class TestVisualizationViewer(unittest.TestCase):
    """Klasa testowa dla przeglądarki wizualizacji"""

    @classmethod
    def setUpClass(cls):
        """Inicjalizacja aplikacji QApplication dla wszystkich testów"""
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication(sys.argv)

    def setUp(self):
        """Tworzenie katalogu tymczasowego z przykładowymi obrazami dla każdego testu"""
        # Utwórz katalog tymczasowy
        self.test_dir = tempfile.mkdtemp(prefix="test_visualizations_")
        
        # Utwórz podfoldery
        self.subfolder1 = os.path.join(self.test_dir, "subfolder1")
        self.subfolder2 = os.path.join(self.test_dir, "subfolder2")
        os.makedirs(self.subfolder1, exist_ok=True)
        os.makedirs(self.subfolder2, exist_ok=True)
        
        # Utwórz przykładowe obrazy testowe
        self._create_test_images()
        
        # Inicjalizuj przeglądarkę
        self.viewer = VisualizationViewer(self.test_dir)
    
    def tearDown(self):
        """Czyszczenie po testach"""
        # Zamknij przeglądarkę
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.close()
        
        # Usuń katalog tymczasowy
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_images(self):
        """Tworzy testowe obrazy PNG używając modułu PIL"""
        try:
            from PIL import Image
            
            # Obraz 1 - czerwony kwadrat
            img1 = Image.new('RGB', (100, 100), color='red')
            img1.save(os.path.join(self.subfolder1, 'red_square.png'))
            
            # Obraz 2 - niebieski kwadrat
            img2 = Image.new('RGB', (100, 100), color='blue')
            img2.save(os.path.join(self.subfolder1, 'blue_square.png'))
            
            # Obraz 3 - zielony kwadrat
            img3 = Image.new('RGB', (100, 100), color='green')
            img3.save(os.path.join(self.subfolder2, 'green_square.png'))
            
            # Obraz 4 - żółty kwadrat
            img4 = Image.new('RGB', (100, 100), color='yellow')
            img4.save(os.path.join(self.subfolder2, 'yellow_square.png'))
            
        except ImportError:
            # Jeśli PIL nie jest dostępny, stwórz puste pliki
            print("UWAGA: Biblioteka PIL nie jest dostępna. Tworzenie pustych plików obrazów.")
            for name in ['red_square.png', 'blue_square.png']:
                with open(os.path.join(self.subfolder1, name), 'w') as f:
                    f.write('Placeholder dla obrazu testowego')
            
            for name in ['green_square.png', 'yellow_square.png']:
                with open(os.path.join(self.subfolder2, name), 'w') as f:
                    f.write('Placeholder dla obrazu testowego')
    
    def test_init(self):
        """Test inicjalizacji przeglądarki"""
        self.assertEqual(self.viewer.results_dir, self.test_dir)
        self.assertEqual(self.viewer.dir_entry.text(), self.test_dir)
        self.assertTrue(len(self.viewer.images_paths) > 0, "Brak załadowanych obrazów")
    
    def test_directory_loading(self):
        """Test ładowania katalogu z wynikami"""
        # Zeruj ścieżki obrazów
        self.viewer.images_paths = []
        
        # Załaduj katalog
        self.viewer.results_dir = self.test_dir
        self.viewer.dir_entry.setText(self.test_dir)
        self.viewer._load_visualizations()
        
        # Sprawdź czy obrazy zostały załadowane
        self.assertTrue(len(self.viewer.images_paths) == 4, f"Nieprawidłowa liczba załadowanych obrazów: {len(self.viewer.images_paths)}")
    
    def test_tree_structure(self):
        """Test struktury drzewa katalogów i plików"""
        # Sprawdź czy drzewo ma 2 elementy główne (2 podfoldery)
        self.assertEqual(self.viewer.tree.topLevelItemCount(), 2, "Nieprawidłowa liczba elementów głównych w drzewie")
        
        # Sprawdź strukturę pierwszego podfolderu
        subfolder1_item = None
        for i in range(self.viewer.tree.topLevelItemCount()):
            item = self.viewer.tree.topLevelItem(i)
            if item.text(0) == "subfolder1":
                subfolder1_item = item
                break
        
        self.assertIsNotNone(subfolder1_item, "Nie znaleziono elementu subfolder1 w drzewie")
        self.assertEqual(subfolder1_item.childCount(), 2, "Nieprawidłowa liczba plików w subfolder1")
    
    def test_image_display(self):
        """Test wyświetlania obrazu"""
        # Sprawdź czy jakiś obraz jest wyświetlany
        self.assertIsNotNone(self.viewer.current_image, "Brak wyświetlanego obrazu")
        
        # Sprawdź czy etykieta obrazu ma ustawioną teksturę
        self.assertFalse(self.viewer.image_label.pixmap().isNull(), "Pixmap obrazu jest pusty")
    
    def test_navigation(self):
        """Test nawigacji pomiędzy obrazami"""
        # Zapamiętaj aktualny obraz
        initial_image = self.viewer.current_image
        
        # Przejdź do następnego obrazu
        self.viewer._next_image()
        next_image = self.viewer.current_image
        
        # Sprawdź czy obraz się zmienił
        self.assertNotEqual(initial_image, next_image, "Nawigacja do następnego obrazu nie działa")
        
        # Przejdź do poprzedniego obrazu
        self.viewer._prev_image()
        prev_image = self.viewer.current_image
        
        # Sprawdź czy wróciliśmy do początkowego obrazu
        self.assertEqual(initial_image, prev_image, "Nawigacja do poprzedniego obrazu nie działa")
    
    def test_filtering(self):
        """Test filtrowania wizualizacji"""
        # Ustaw filtr na 'red'
        self.viewer.filter_entry.setText("red")
        self.viewer._apply_filter()
        
        # Sprawdź czy tylko czerwony obraz jest widoczny
        visible_count = 0
        for i in range(self.viewer.tree.topLevelItemCount()):
            top_item = self.viewer.tree.topLevelItem(i)
            if not top_item.isHidden():
                for j in range(top_item.childCount()):
                    child = top_item.child(j)
                    if not child.isHidden():
                        visible_count += 1
        
        self.assertEqual(visible_count, 1, f"Nieprawidłowa liczba widocznych elementów po filtrowaniu: {visible_count}")
        
        # Wyczyść filtr
        self.viewer._clear_filter()
        
        # Sprawdź czy wszystkie obrazy są znowu widoczne
        visible_count = 0
        for i in range(self.viewer.tree.topLevelItemCount()):
            top_item = self.viewer.tree.topLevelItem(i)
            if not top_item.isHidden():
                for j in range(top_item.childCount()):
                    child = top_item.child(j)
                    if not child.isHidden():
                        visible_count += 1
        
        self.assertEqual(visible_count, 4, f"Nieprawidłowa liczba widocznych elementów po wyczyszczeniu filtra: {visible_count}")


class TestVisualizationViewerGUI(unittest.TestCase):
    """Klasa do testowania GUI przeglądarki wizualizacji"""
    
    @classmethod
    def setUpClass(cls):
        """Inicjalizacja dla testów GUI"""
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """Tworzenie katalogu tymczasowego z przykładowymi obrazami"""
        # Utwórz katalog tymczasowy
        self.test_dir = tempfile.mkdtemp(prefix="test_gui_")
        
        # Utwórz podfoldery
        self.subfolder = os.path.join(self.test_dir, "test_folder")
        os.makedirs(self.subfolder, exist_ok=True)
        
        # Utwórz przykładowe obrazy
        self._create_test_images()
        
        # Inicjalizuj przeglądarkę
        self.viewer = VisualizationViewer(self.test_dir)
        self.viewer.show()
    
    def tearDown(self):
        """Czyszczenie po testach"""
        if hasattr(self, "viewer") and self.viewer:
            self.viewer.close()
        
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_images(self):
        """Tworzy testowe obrazy PNG"""
        try:
            from PIL import Image
            
            # Obraz 1
            img1 = Image.new('RGB', (200, 200), color='red')
            img1.save(os.path.join(self.subfolder, 'test_image1.png'))
            
            # Obraz 2
            img2 = Image.new('RGB', (200, 200), color='blue')
            img2.save(os.path.join(self.subfolder, 'test_image2.png'))
            
        except ImportError:
            print("UWAGA: Biblioteka PIL nie jest dostępna. Tworzenie pustych plików.")
            for name in ['test_image1.png', 'test_image2.png']:
                with open(os.path.join(self.subfolder, name), 'w') as f:
                    f.write('Placeholder dla obrazu testowego')
    
    def test_button_clicks(self):
        """Test kliknięć przycisków"""
        # Zapamiętaj początkowy obraz
        initial_image = self.viewer.current_image
        
        # Kliknij przycisk "Następny"
        next_button = None
        for child in self.viewer.findChildren(QPushButton):
            if "Następny" in child.text():
                next_button = child
                break
        
        self.assertIsNotNone(next_button, "Nie znaleziono przycisku 'Następny'")
        QTest.mouseClick(next_button, Qt.LeftButton)
        
        # Poczekaj na przetworzenie zdarzeń
        QTest.qWait(500)
        
        # Sprawdź czy obraz się zmienił
        self.assertNotEqual(initial_image, self.viewer.current_image, "Przycisk 'Następny' nie działa")
    
    def test_directory_selection(self):
        """Test wyboru katalogu"""
        # Utwórz nowy katalog tymczasowy
        new_dir = tempfile.mkdtemp(prefix="test_new_dir_")
        try:
            # Symuluj wybór katalogu
            self.viewer.results_dir = new_dir
            self.viewer.dir_entry.setText(new_dir)
            
            # Kliknij przycisk "Załaduj wizualizacje"
            load_button = None
            for child in self.viewer.findChildren(QPushButton):
                if "Załaduj" in child.text():
                    load_button = child
                    break
            
            self.assertIsNotNone(load_button, "Nie znaleziono przycisku 'Załaduj wizualizacje'")
            
            # Utwórz czasomierz do zamknięcia komunikatu o braku wizualizacji
            def close_messagebox():
                for widget in QApplication.topLevelWidgets():
                    if widget.isVisible() and widget.inherits('QMessageBox'):
                        QTest.keyClick(widget, Qt.Key_Return)
            
            # Zaplanuj zamknięcie komunikatu
            QTimer.singleShot(1000, close_messagebox)
            
            # Kliknij przycisk
            QTest.mouseClick(load_button, Qt.LeftButton)
            
            # Poczekaj na przetworzenie zdarzeń
            QTest.qWait(2000)
            
            # Sprawdź czy katalog został zaktualizowany
            self.assertEqual(self.viewer.results_dir, new_dir, "Katalog nie został zaktualizowany")
            
        finally:
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)


def create_sample_visualizations():
    """Tworzy przykładowe wizualizacje w folderze wyniki_test"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import math
        
        # Utwórz katalog dla wyników
        output_dir = "wyniki_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # Utwórz podfoldery dla różnych typów wizualizacji
        histogram_dir = os.path.join(output_dir, "histogramy")
        plot_dir = os.path.join(output_dir, "wykresy")
        heatmap_dir = os.path.join(output_dir, "mapy_ciepla")
        
        os.makedirs(histogram_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Funkcja do rysowania histogramu
        def create_histogram(filename, title, values, color):
            img = Image.new('RGB', (500, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            # Rysuj słupki
            max_value = max(values)
            bar_width = 30
            gap = 10
            
            for i, value in enumerate(values):
                bar_height = int((value / max_value) * 300)
                x0 = 50 + i * (bar_width + gap)
                y0 = 350 - bar_height
                x1 = x0 + bar_width
                y1 = 350
                draw.rectangle([x0, y0, x1, y1], fill=color, outline='black')
                draw.text((x0 + bar_width/2 - 3, y1 + 10), str(i), fill='black')
            
            # Rysuj etykiety
            draw.text((200, 20), title, fill='black')
            draw.line([(50, 350), (450, 350)], fill='black', width=2)  # oś X
            draw.line([(50, 50), (50, 350)], fill='black', width=2)    # oś Y
            
            img.save(filename)
        
        # Funkcja do rysowania wykresu
        def create_plot(filename, title, x_values, y_values, color):
            img = Image.new('RGB', (500, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            # Rysuj punkty i linie
            points = []
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            
            for i in range(len(x_values)):
                # Skalowanie do współrzędnych obrazu
                x = 50 + int((x_values[i] - min_x) / (max_x - min_x) * 400)
                y = 350 - int((y_values[i] - min_y) / (max_y - min_y) * 300)
                points.append((x, y))
                draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=color)
            
            # Rysuj linie między punktami
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill=color, width=2)
            
            # Rysuj etykiety
            draw.text((200, 20), title, fill='black')
            draw.line([(50, 350), (450, 350)], fill='black', width=2)  # oś X
            draw.line([(50, 50), (50, 350)], fill='black', width=2)    # oś Y
            
            img.save(filename)
        
        # Funkcja do rysowania mapy ciepła
        def create_heatmap(filename, title, data, colormap='hot'):
            img = Image.new('RGB', (500, 400), color='white')
            draw = ImageDraw.Draw(img)
            
            # Rozmiary mapy ciepła
            cell_size = 20
            rows, cols = data.shape
            
            # Rysuj komórki
            for i in range(rows):
                for j in range(cols):
                    # Normalizacja wartości do zakresu 0-1
                    value = (data[i, j] - data.min()) / (data.max() - data.min())
                    
                    # Mapa kolorów (uproszczona)
                    if colormap == 'hot':
                        r = int(255 * min(value * 2, 1))
                        g = int(255 * min(value * 2 - 1, 1) if value > 0.5 else 0)
                        b = int(255 * min(value * 2 - 1.5, 1) if value > 0.75 else 0)
                        color = (r, g, b)
                    else:  # mapa niebiesko-czerwona
                        r = int(255 * value)
                        b = int(255 * (1 - value))
                        g = int(255 * (1 - abs(2 * value - 1)))
                        color = (r, g, b)
                    
                    x0 = 50 + j * cell_size
                    y0 = 50 + i * cell_size
                    x1 = x0 + cell_size
                    y1 = y0 + cell_size
                    draw.rectangle([x0, y0, x1, y1], fill=color, outline='black')
            
            # Rysuj etykiety
            draw.text((200, 20), title, fill='black')
            
            img.save(filename)
        
        # Generuj przykładowe dane
        np.random.seed(42)
        
        # Dane dla histogramów
        hist_data1 = np.random.normal(50, 10, 10).astype(int)
        hist_data2 = np.random.poisson(5, 10)
        
        # Dane dla wykresów
        x_data1 = np.linspace(0, 10, 20)
        y_data1 = np.sin(x_data1)
        
        x_data2 = np.linspace(0, 10, 20)
        y_data2 = x_data2 ** 2 / 10
        
        # Dane dla map ciepła
        heatmap_data1 = np.random.rand(10, 10)
        heatmap_data2 = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                heatmap_data2[i, j] = math.sin(i/5) * math.cos(j/5)
        
        # Generuj wizualizacje
        create_histogram(os.path.join(histogram_dir, "histogram_normalny.png"), 
                        "Rozkład normalny", hist_data1, "blue")
        create_histogram(os.path.join(histogram_dir, "histogram_poisson.png"), 
                        "Rozkład Poissona", hist_data2, "green")
        
        create_plot(os.path.join(plot_dir, "wykres_sin.png"), 
                  "Funkcja sinus", x_data1, y_data1, "red")
        create_plot(os.path.join(plot_dir, "wykres_kwadratowy.png"), 
                  "Funkcja kwadratowa", x_data2, y_data2, "purple")
        
        create_heatmap(os.path.join(heatmap_dir, "mapa_ciepla_losowa.png"), 
                      "Losowa mapa ciepła", heatmap_data1, "hot")
        create_heatmap(os.path.join(heatmap_dir, "mapa_ciepla_sincos.png"), 
                      "Mapa ciepła sin*cos", heatmap_data2, "coolwarm")
        
        print(f"Utworzono przykładowe wizualizacje w katalogu {output_dir}")
        return output_dir
        
    except ImportError:
        print("UWAGA: Biblioteka PIL lub NumPy nie są dostępne. Nie można wygenerować przykładowych wizualizacji.")
        return None


def main():
    """Główna funkcja testowa"""
    # Sprawdź czy użytkownik chce wygenerować przykładowe dane
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-samples':
        create_sample_visualizations()
        return
    
    # Uruchom testy
    print("Uruchamianie testów jednostkowych...")
    unittest.main(argv=['first-arg-is-ignored'])


if __name__ == "__main__":
    main() 