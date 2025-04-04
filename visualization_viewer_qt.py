#!/usr/bin/env python3
import os
import sys
import glob
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTreeWidget, QTreeWidgetItem, QSplitter, QLineEdit,
                             QPushButton, QFileDialog, QMessageBox, QScrollArea)
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QSize

class VisualizationViewer(QMainWindow):
    def __init__(self, results_dir=None):
        super().__init__()
        self.setWindowTitle("Przeglądarka Wizualizacji")
        self.resize(1200, 700)
        
        # Zmienne dla wybranego katalogu z wynikami i aktualnego obrazu
        self.results_dir = results_dir
        self.current_image = None
        self.images_paths = []
        self.current_index = 0
        
        # Tworzenie interfejsu
        self._create_ui()
        
        # Jeśli został podany katalog, załaduj wizualizacje
        if self.results_dir:
            self.dir_entry.setText(self.results_dir)
            self._load_visualizations()
    
    def _create_ui(self):
        """Tworzy interfejs użytkownika"""
        # Główny widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Główny układ
        main_layout = QVBoxLayout(central_widget)
        
        # Nagłówek
        header_label = QLabel("Przeglądarka Wizualizacji", self)
        header_label.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 14, QFont.Bold)
        header_label.setFont(font)
        main_layout.addWidget(header_label)
        
        # Panel wyboru katalogu
        dir_layout = QHBoxLayout()
        dir_label = QLabel("Katalog z wynikami:", self)
        dir_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.dir_entry = QLineEdit(self)
        self.dir_entry.setMinimumWidth(400)
        
        select_btn = QPushButton("Wybierz katalog", self)
        select_btn.clicked.connect(self._select_directory)
        
        load_btn = QPushButton("Załaduj wizualizacje", self)
        load_btn.clicked.connect(self._load_visualizations)
        
        dir_layout.addWidget(dir_label)
        dir_layout.addWidget(self.dir_entry)
        dir_layout.addWidget(select_btn)
        dir_layout.addWidget(load_btn)
        main_layout.addLayout(dir_layout)
        
        # Panel filtrowania
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filtruj wizualizacje:", self)
        filter_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.filter_entry = QLineEdit(self)
        
        filter_btn = QPushButton("Zastosuj filtr", self)
        filter_btn.clicked.connect(self._apply_filter)
        
        clear_btn = QPushButton("Wyczyść filtr", self)
        clear_btn.clicked.connect(self._clear_filter)
        
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_entry)
        filter_layout.addWidget(filter_btn)
        filter_layout.addWidget(clear_btn)
        main_layout.addLayout(filter_layout)
        
        # Splitter do podziału okna na listę plików i podgląd
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel listy plików
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_label = QLabel("Lista wizualizacji")
        list_label.setFont(QFont("Arial", 10, QFont.Bold))
        list_layout.addWidget(list_label)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Wizualizacje")
        self.tree.setMinimumWidth(300)
        self.tree.itemClicked.connect(self._on_item_clicked)
        list_layout.addWidget(self.tree)
        
        # Panel podglądu obrazu
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        
        preview_label = QLabel("Podgląd wizualizacji")
        preview_label.setFont(QFont("Arial", 10, QFont.Bold))
        preview_layout.addWidget(preview_label)
        
        self.image_info = QLabel("")
        self.image_info.setFont(QFont("Arial", 10))
        preview_layout.addWidget(self.image_info)
        
        # Scroll Area dla obrazu
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        
        scroll_area.setWidget(self.image_label)
        preview_layout.addWidget(scroll_area)
        
        # Dodaj oba panele do splittera
        splitter.addWidget(list_widget)
        splitter.addWidget(preview_widget)
        splitter.setSizes([300, 900])  # Proporcje 1:3
        
        main_layout.addWidget(splitter)
        
        # Panel nawigacji
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("⬅️ Poprzedni", self)
        prev_btn.clicked.connect(self._prev_image)
        
        next_btn = QPushButton("Następny ➡️", self)
        next_btn.clicked.connect(self._next_image)
        
        open_btn = QPushButton("🔍 Otwórz w programie", self)
        open_btn.clicked.connect(self._open_externally)
        
        self.info_label = QLabel("")
        self.info_label.setFont(QFont("Arial", 10, QFont.Bold))
        
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        nav_layout.addWidget(open_btn)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.info_label)
        
        main_layout.addLayout(nav_layout)
        
        # Panel z instrukcją
        help_label = QLabel("📋 Instrukcja: Wybierz katalog z wynikami, a następnie kliknij na wizualizację, aby ją wyświetlić. Użyj przycisków 'Poprzedni' i 'Następny' do nawigacji.")
        help_label.setWordWrap(True)
        main_layout.addWidget(help_label)
        
        # Panel statusu
        self.status_label = QLabel("Status: Gotowy")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        main_layout.addWidget(self.status_label)
    
    def _select_directory(self):
        """Wybór katalogu z wynikami"""
        directory = QFileDialog.getExistingDirectory(
            self, "Wybierz katalog z wynikami", "", QFileDialog.ShowDirsOnly
        )
        if directory:
            self.results_dir = directory
            self.dir_entry.setText(directory)
            self._load_visualizations()
    
    def _load_visualizations(self):
        """Ładuje wizualizacje z katalogu wyników"""
        try:
            if not self.results_dir or not os.path.exists(self.results_dir):
                QMessageBox.critical(self, "Błąd", "Wybierz prawidłowy katalog z wynikami!")
                return
            
            self._set_status(f"Ładowanie wizualizacji z {self.results_dir}", "blue")
            
            # Wyczyść stare dane
            self.tree.clear()
            self.images_paths = []
            self.current_index = 0
            
            # Znajdź wszystkie pliki PNG
            all_images = []
            for root, dirs, files in os.walk(self.results_dir):
                for file in files:
                    if file.endswith('.png'):
                        all_images.append(os.path.join(root, file))
            
            if not all_images:
                QMessageBox.information(self, "Informacja", "Nie znaleziono wizualizacji w wybranym katalogu!")
                self._set_status("Nie znaleziono wizualizacji", "red")
                return
            
            # Grupowanie obrazów według katalogów
            image_dirs = {}
            for img_path in all_images:
                dir_name = os.path.dirname(img_path)
                dir_name_short = os.path.basename(dir_name)
                
                if dir_name not in image_dirs:
                    dir_item = QTreeWidgetItem(self.tree, [dir_name_short])
                    dir_item.setExpanded(True)
                    image_dirs[dir_name] = dir_item
                
                # Dodaj obraz do drzewa
                file_name = os.path.basename(img_path)
                item = QTreeWidgetItem(image_dirs[dir_name], [file_name])
                item.setData(0, Qt.UserRole, img_path)
                
                # Dodaj ścieżkę do listy
                self.images_paths.append(img_path)
            
            # Zaktualizuj informację o liczbie obrazów
            self.info_label.setText(f"📊 Znaleziono {len(all_images)} wizualizacji")
            
            # Wyświetl pierwszy obraz
            if self.images_paths:
                self.current_index = 0
                self._display_image(self.images_paths[0])
                
            self._set_status(f"Załadowano {len(all_images)} wizualizacji", "green")
        except Exception as e:
            self._set_status(f"Błąd: {str(e)}", "red")
            QMessageBox.critical(self, "Błąd", f"Wystąpił błąd podczas ładowania wizualizacji: {str(e)}")
    
    def _apply_filter(self):
        """Filtruje listę wizualizacji"""
        filter_text = self.filter_entry.text().lower()
        if not filter_text:
            return
        
        try:
            # Ukryj elementy, które nie pasują do filtra
            for i in range(self.tree.topLevelItemCount()):
                dir_item = self.tree.topLevelItem(i)
                show_dir = False
                
                for j in range(dir_item.childCount()):
                    child_item = dir_item.child(j)
                    item_text = child_item.text(0).lower()
                    
                    if filter_text in item_text:
                        child_item.setHidden(False)
                        show_dir = True
                    else:
                        child_item.setHidden(True)
                
                dir_item.setHidden(not show_dir)
                
            self._set_status(f"Zastosowano filtr '{filter_text}'", "green")
        except Exception as e:
            self._set_status(f"Błąd filtru: {str(e)}", "red")
    
    def _clear_filter(self):
        """Czyści filtr i przywraca wszystkie obrazy"""
        self.filter_entry.clear()
        self._load_visualizations()
        self._set_status("Filtr wyczyszczony", "green")
    
    def _on_item_clicked(self, item, column):
        """Obsługuje kliknięcie elementu drzewa"""
        try:
            # Sprawdź czy to element z obrazem (nie katalog)
            img_path = item.data(0, Qt.UserRole)
            if img_path:
                self._display_image(img_path)
                
                # Aktualizacja indeksu
                if img_path in self.images_paths:
                    self.current_index = self.images_paths.index(img_path)
                
                self._set_status(f"Wybrano {os.path.basename(img_path)}", "green")
        except Exception as e:
            self._set_status(f"Błąd wyboru: {str(e)}", "red")
    
    def _display_image(self, img_path):
        """Wyświetla obraz"""
        if not img_path or not os.path.exists(img_path):
            self._set_status(f"Błąd: Nie znaleziono pliku {img_path}", "red")
            return
        
        try:
            # Załaduj obraz przy użyciu QPixmap
            pixmap = QPixmap(img_path)
            
            if pixmap.isNull():
                self._set_status(f"Błąd: Nie można wczytać obrazu {img_path}", "red")
                return
            
            # Pobierz oryginalne wymiary obrazu
            img_width = pixmap.width()
            img_height = pixmap.height()
            
            # Dodaj informację o obrazie
            file_name = os.path.basename(img_path)
            dir_name = os.path.basename(os.path.dirname(img_path))
            self.image_info.setText(f"🖼️ Wizualizacja: {file_name} ({img_width}x{img_height}) z {dir_name}")
            
            # Dopasuj obraz do okna z zachowaniem proporcji
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Ustaw aktualny obraz
            self.current_image = img_path
            
            # Zaktualizuj etykietę informacyjną w dolnym pasku
            self.info_label.setText(f"📊 Wizualizacja {self.current_index + 1} z {len(self.images_paths)}")
            
            self._set_status(f"Wyświetlono {file_name}", "green")
        except Exception as e:
            self._set_status(f"Błąd wyświetlania: {str(e)}", "red")
            QMessageBox.critical(self, "Błąd", f"Nie udało się wyświetlić obrazu: {str(e)}")
    
    def _prev_image(self):
        """Przechodzi do poprzedniego obrazu"""
        if not self.images_paths:
            return
        
        self.current_index = (self.current_index - 1) % len(self.images_paths)
        self._display_image(self.images_paths[self.current_index])
    
    def _next_image(self):
        """Przechodzi do następnego obrazu"""
        if not self.images_paths:
            return
        
        self.current_index = (self.current_index + 1) % len(self.images_paths)
        self._display_image(self.images_paths[self.current_index])
    
    def _open_externally(self):
        """Otwiera aktualny obraz w zewnętrznym programie"""
        if not self.current_image:
            QMessageBox.information(self, "Informacja", "Najpierw wybierz obraz do otwarcia.")
            return
        
        try:
            import subprocess
            
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', self.current_image])
            elif sys.platform == 'win32':  # Windows
                os.startfile(self.current_image)
            else:  # Linux
                subprocess.run(['xdg-open', self.current_image])
                
            self._set_status(f"Otwarto {os.path.basename(self.current_image)} w zewnętrznym programie", "green")
        except Exception as e:
            self._set_status(f"Błąd otwierania: {str(e)}", "red")
            QMessageBox.critical(self, "Błąd", f"Nie udało się otworzyć obrazu: {str(e)}")
    
    def _set_status(self, message, color):
        """Ustawia komunikat statusu z określonym kolorem"""
        self.status_label.setText(f"Status: {message}")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
    
    def resizeEvent(self, event):
        """Obsługa zmiany rozmiaru okna"""
        super().resizeEvent(event)
        # Odśwież obraz przy zmianie rozmiaru
        if self.current_image:
            self._display_image(self.current_image)


def main():
    app = QApplication(sys.argv)
    
    # Sprawdzanie czy podano katalog jako argument
    results_dir = None
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Znajdź najnowszy katalog wyników
        result_dirs = glob.glob("wyniki_*")
        if result_dirs:
            results_dir = max(result_dirs, key=os.path.getctime)
    
    # Utwórz i pokaż okno główne
    window = VisualizationViewer(results_dir)
    window.show()
    
    # Uruchom pętlę zdarzeń aplikacji
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 