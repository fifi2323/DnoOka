import tkinter as tk
from tkinter import ttk, filedialog, Label, Button, Canvas, Checkbutton, IntVar, DoubleVar, StringVar, Entry, simpledialog
from PIL import Image, ImageTk
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import time
import joblib
from skimage.feature import hog
from skimage.measure import moments_central, moments_hu
from skimage.filters import frangi
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2

class DnoOka:
    def __init__(self, root):
        self.predicted_image = None
        self.root = root
        self.root.title("Symulator Tomografu Komputerowego")

        # Wczytanie modelu
        try:
            self.clf = joblib.load(".venv/random_forest_model.pkl")
        except FileNotFoundError:
            print("Nie znaleziono modelu! Upewnij się, że plik modelu jest w odpowiedniej lokalizacji.")


        # Ramka na wczytywanie obrazu i parametry
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Ramka do wyświetlania obrazów
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Ramka do wyświetlania predykcji
        self.image_frame_pred = tk.Frame(root)
        self.image_frame_pred.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Przycisk do wczytywania obrazu
        self.load_button = Button(self.control_frame, text="Wczytaj obraz", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        # Przycisk do generowania predykcji
        self.pred_button = Button(self.control_frame, text="Znajdź naczynia", command=self.processing_method_choice)
        self.pred_button.grid(row=0, column=1, padx=5, pady=5)

        # Przycisk do zapisywania obrazu
        self.save_button = Button(self.control_frame, text="Zapisz obraz", command=self.save_image)
        self.save_button.grid(row=0, column=2, padx=5, pady=5)

        # Wybór sposobu przetwarzania
        # Etykieta obok listy
        tk.Label(self.control_frame, text="Wybór sposobu przetwarzania:").grid(row=1, column=0, padx=5, pady=5)

        # Zmienna przechowująca wybraną wartość (string)
        self.processing_var = tk.StringVar()

        # Definiujemy Combobox z czterema opcjami
        self.filter_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.processing_var,
            values=["1 - filtr  Frangi’ego - 3.0", "2 - gotowy klasyfiklator - 4.0", "3 - głęboka sieć neuronowa - 5.0"],
            state="readonly"
        )
        self.filter_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.filter_combobox.current(0)  # Ustawiamy domyślnie pierwszą opcję ("0 - brak")

        # Canvas do wyświetlania oryginalnego obrazu
        self.original_picture_canvas = Canvas(self.image_frame, width=400, height=400, bg="black")
        self.original_picture_canvas.grid(row=0, column=0, padx=15, pady=10)
        Label(self.image_frame, text="Oryginalny obraz").grid(row=2, column=0)

        # Canvas do wyświetlania obrazu po predykcji
        self.prediction_picture_canvas = Canvas(self.image_frame_pred, width=400, height=400, bg="black")
        self.prediction_picture_canvas.grid(row=0, column=0, padx=15, pady=10)
        Label(self.image_frame_pred, text="Obraz po predykcji").grid(row=3, column=0)

        self.image = None
        self.image_array = None
        self.reconstructed_image = None

    def processing_method_choice(self):
        selected_method = self.processing_var.get()
        selected_method_num = int(selected_method.split("-")[0].strip())

        if selected_method_num == 1:
            self.show_frangi_result()
        elif selected_method == 2:
            self.predict()

        return


    def preprocess_image(self, image_array):
        """
        Wstępne przetwarzanie obrazu:
          - Konwersja do skali szarości.
          - Rozmycie Gaussa (redukcja szumu).
          - Normalizacja histogramu (wyrównanie kontrastu).
          - Wyostrzenie obrazu.
        """

        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        eq = cv2.equalizeHist(blurred)

        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(eq, -1, kernel_sharpen)
        return sharpened

    def frangi_extract_vessels(self, preprocessed_image):
        """
        Właściwe przetwarzanie obrazu:
         - Zastosowanie filtru Frangi'ego do wykrycia naczyń.
        """
        normalized = preprocessed_image / 255.0
        vessel_enhanced = frangi(normalized)
        return vessel_enhanced

    def frangi_postprocess_vessel_image(self, vessel_image, threshold=0.2):
        """
        Końcowe przetwarzanie obrazu:
          - Progowanie wzmocnionego obrazu w celu uzyskania binarnej maski.
          - Operacje morfologiczne (otwarcie i zamknięcie) dla redukcji szumów.
        Zwraca binarną maskę wykrytych naczyń.
        """
        binary_mask = (vessel_image > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def show_frangi_result(self):
        # Wywołanie pipeline'u filtru Frangi – poprawne nazwy funkcji
        preprocessed = self.preprocess_image(self.image_array)
        vessel_enhanced = self.frangi_extract_vessels(preprocessed)
        mask = self.frangi_postprocess_vessel_image(vessel_enhanced, threshold=0.01)

        # Konwersja do formatu PIL
        result_image = Image.fromarray(mask)

        # Skalowanie do rozmiaru canvasa
        max_canvas_size = 400
        img_width, img_height = result_image.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        result_resized = result_image.resize(new_size)

        # Konwersja na PhotoImage i aktualizacja canvasa
        self.predicted_image = ImageTk.PhotoImage(result_resized)
        self.prediction_picture_canvas.delete("all")
        canvas_width = self.prediction_picture_canvas.winfo_width()
        canvas_height = self.prediction_picture_canvas.winfo_height()
        self.prediction_picture_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                                    image=self.predicted_image, anchor=tk.CENTER)

    def extract_features(self, image):
        """ Ekstrakcja cech z wycinka obrazu """
        # Wariancja kolorów
        variance_r = np.var(image[:, :, 0])
        variance_g = np.var(image[:, :, 1])
        variance_b = np.var(image[:, :, 2])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Momenty centralne - sprawdzamy, czy moment zerowy nie jest równy 0
        moments = moments_central(gray)
        if moments[0, 0] == 0:
            central_moment = 0
        else:
            central_moment = moments[0, 2] + moments[2, 0]

        # Momenty Hu - dodajemy zabezpieczenie przed log(0) i NaN
        hu_moments = moments_hu(moments)
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        hu_moments = np.nan_to_num(hu_moments, nan=0.0, posinf=0.0, neginf=0.0)

        # Histogram gradientów (HOG)
        hog_features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)

        # Połączenie cech w jeden wektor
        features = [variance_r, variance_g, variance_b, central_moment] + list(hu_moments) + list(hog_features[:10])

        return features

    def predict(self):
        """ Przewidywanie na podstawie modelu i wyświetlanie wyników """
        if self.image_array is None:
            return

        # Rozdzielamy obraz na fragmenty i wyciągamy cechy
        window_size = 8
        features = []
        print(self.image_array.shape)
        for i in range(4, self.image_array.shape[0] - 4, window_size):
            for j in range(4, self.image_array.shape[1] - 4, window_size):
                feature = self.extract_features(self.image_array[i - 4:i + 4, j - 4:j + 4])  # 8x8 okno
                features.append(feature)

        features = np.array(features)

        # Predykcja
        print(features.shape)
        predictions = self.clf.predict(features)
        print("Unikalne wartości predykcji:", np.unique(predictions))


        # Przekształcenie predictions do uint8 przed użyciem
        predictions = predictions.astype(np.uint8)

        # Inicjalizacja obrazu predykcji (bez wartości w 3 kanałach kolorów)
        prediction_image = np.zeros(self.image_array.shape[:2], dtype=np.uint8)

        index = 0
        for i in range(4, self.image_array.shape[0] - 4, window_size):
            for j in range(4, self.image_array.shape[1] - 4, window_size):
                prediction_image[i:i + window_size, j:j + window_size] = predictions[index]
                index += 1

        # Konwersja do formatu PIL
        self.predicted_image_pil = Image.fromarray(prediction_image)

        # Dopasowanie rozmiaru obrazu do wyświetlania
        max_canvas_size = 400
        img_width, img_height = self.predicted_image_pil.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        predicted_display_image = self.predicted_image_pil.resize(new_size)
        self.predicted_image = ImageTk.PhotoImage(predicted_display_image)

        # Upewnienie się, że Canvas ma poprawne wymiary
        self.root.update()
        canvas_width = self.prediction_picture_canvas.winfo_width()
        canvas_height = self.prediction_picture_canvas.winfo_height()

        self.prediction_picture_canvas.delete("all")
        self.prediction_picture_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                                    image=self.predicted_image, anchor=tk.CENTER)
    def save_image(self):  # Convert NumPy array to PIL Image
        self.predicted_image_pil.save('plik.png')  # Save the image
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp;*.dcm")])
        if not file_path:
            return
        image = Image.open(file_path)
        self.image_array = np.array(image)
        self.image = ImageTk.PhotoImage(image)


        # żeby poprawnie wyświetlić obraz w Canvas jest on skalowany - jest to kopia niewykorzysywana do obliczeń
        max_canvas_size = 400
        img_width, img_height = image.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        display_image = image.resize(new_size)  # Skopiowana wersja do wyświetlania
        self.image = ImageTk.PhotoImage(display_image)

        # Wyśrodkowanie obrazu na Canvas
        self.original_picture_canvas.delete("all")
        self.original_picture_canvas.create_image(200, 200, image=self.image, anchor=tk.CENTER)


        self.pred_button.config(state=tk.NORMAL)












if __name__ == "__main__":
    root = tk.Tk()
    app = DnoOka(root)
    root.mainloop()
