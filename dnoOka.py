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
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import remove_small_objects
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from skimage.morphology import opening, closing, footprint_rectangle
from tensorflow.keras.models import load_model
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

        try:
            self.unet_model = load_model('unet_retinal_vessel.h5')  # Load the U-Net model
        except Exception as e:
            print(f"Error loading U-Net model: {e}")
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
            values=["1 - filtr  Frangi’ego - 3.0", "2 - gotowy klasyfiklator - 4.0", "3 - głęboka sieć neuronowa - 5.0", "4 - Odszumianie gotowego obrazu - cos"],
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
        elif selected_method_num == 2:
            self.predict()
        elif selected_method_num == 3:
            self.unet_predict()
        elif selected_method_num == 4:
            self.pure_postprocessing(self.image_array)
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

        denoised = denoise_tv_chambolle(eq, weight=0.1)


        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
        return sharpened

    def frangi_extract_vessels(self, preprocessed_image):
        """
        Właściwe przetwarzanie obrazu:
         - Zastosowanie filtru Frangi'ego do wykrycia naczyń.
        """
        normalized = preprocessed_image / 255.0
        vessel_enhanced = frangi(normalized)
        return vessel_enhanced

    def frangi_postprocess_vessel_image(self, vessel_image, threshold):
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
        start = time.time()
        # Wywołanie pipeline'u filtru Frangi – poprawne nazwy funkcji
        preprocessed = self.preprocess_image(self.image_array)

        vessel_enhanced = self.frangi_extract_vessels(preprocessed)

        mask = self.frangi_postprocess_vessel_image(vessel_enhanced, threshold=0.001)
        self.print_result(mask)

        np_image_8bit = (mask * 255).astype(np.uint8)
        self.predicted_image_pil = Image.fromarray(mask)
        stop = time.time()
        print(f"Czas przetwarzania: {stop - start}")



    def print_result(self, picture):
        # Konwersja do formatu PIL
        result_image = Image.fromarray(picture)

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
        self.prediction_picture_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
        time.sleep(0.01)  # Opóźnienie

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
       # hog_features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)

        # Połączenie cech w jeden wektor
        features = [variance_r, variance_g, variance_b, central_moment] + list(hu_moments)

        return features

    def pure_postprocessing(self, prediction_image):
        """ Usuwanie szumu z obrazu predykcji """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(prediction_image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


        # Konwersja do formatu PIL
        self.predicted_image_pil = Image.fromarray(closed)

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
        return
    def random_forest_postprocessing(self, prediction_image):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(prediction_image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed
    def predict(self):
        """ Przewidywanie na podstawie modelu i wyświetlanie wyników """
        if self.image_array is None:
            return
        start = time.time()
        # Rozdzielamy obraz na fragmenty i wyciągamy cechy
        window_size = 8
        features = []
        indices = []
        print(self.image_array.shape)
        for i in range(3, self.image_array.shape[0] - 3, 2):
            for j in range(3, self.image_array.shape[1] - 3, 2):
                if sum(self.image_array[i, j, :]) != 0:
                    feature = self.extract_features(self.image_array[i - 3:i + 3, j - 3:j + 3])  # 8x8 okno
                    features.append(feature)
                    indices.append([i, j])
            print(i)
        features = np.array(features)

        # Predykcja
        print(features.shape)
        predictions = self.clf.predict(features)
        print("Unikalne wartości predykcji:", np.unique(predictions))

        # Przekształcenie predictions do uint8 przed użyciem
        predictions = predictions.astype(np.uint8)

        # Inicjalizacja obrazu predykcji
        prediction_image = np.zeros(self.image_array.shape[:2], dtype=np.uint8)

        for index, prediction in zip(indices, predictions):
            try:
                i =index[0]
                j = index[1]
                prediction_image[i:i + 2, j:j + 2] = prediction
            except IndexError:
                pass

        # Postprocessing - usunięcie szumu
        prediction_image = self.random_forest_postprocessing(prediction_image)

        stop = time.time()
        print(f"Czas przetwarzania: {stop - start}")

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
        # Pobierz nazwę pliku do zapisu
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Image Files", "*.png;*.jpg;*.bmp;*.dcm")],
            title="Zapisz plik"
        )
        if not filename:
            return
        self.predicted_image_pil.save(filename)  # Save the image
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

    def unet_predict(self):
        """ U-Net Prediction Method with Patch-based Processing """

        img_patches, original_size, padded_size = self.preprocess_unet_image(self.image_array)  # Get image patches
        patch_size = 256

        # Predict masks for each patch
        pred_patches = []
        n = len(img_patches)
        for i, patch in enumerate(img_patches):
            print(f"Patch {i} / {n}")
            patch = np.expand_dims(patch, axis=0)  # Add batch dimension
            pred_mask = self.unet_model.predict(patch)[0]  # Remove batch dimension
            pred_mask = np.squeeze(pred_mask)  # Ensure it's 2D
            print(pred_mask.min(), pred_mask.max())
            pred_mask = (pred_mask > 0.1).astype(np.uint8)  # Thresholding
            pred_patches.append(pred_mask)

        # Reconstruct the full-size image from patches
        h, w = padded_size
        full_mask = np.zeros((h, w), dtype=np.uint8)

        idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                full_mask[i:i + patch_size, j:j + patch_size] = pred_patches[idx]
                idx += 1
        print(full_mask)
        print(full_mask.min(), full_mask.max())
        print(sum(full_mask))
        # Crop back to original image size
        full_mask = full_mask[:original_size[0], :original_size[1]]

        # Convert to PIL image (scale to 0-255 and convert to grayscale)
        self.predicted_image_pil = Image.fromarray(full_mask * 255).convert('L')

        # Resize for display
        max_canvas_size = 400
        img_width, img_height = self.predicted_image_pil.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        predicted_display_image = self.predicted_image_pil.resize(new_size)

        # Convert to Tkinter PhotoImage
        self.predicted_image = ImageTk.PhotoImage(predicted_display_image)

        # Update canvas dimensions
        self.prediction_picture_canvas.config(width=new_size[0], height=new_size[1])
        self.root.update()

        # Display image on canvas
        self.prediction_picture_canvas.delete("all")
        self.prediction_picture_canvas.create_image(
            new_size[0] // 2, new_size[1] // 2,  # Center image
            image=self.predicted_image, anchor=tk.CENTER
        )

    def preprocess_unet_image(self, image_array, img_size=(256, 256)):
        """ Preprocess Image: Padding & Splitting into Patches """
        # Convert image to grayscale if it has 3 channels (RGB)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        h, w = image_array.shape[:2]

        # Pad image to be a multiple of 256
        pad_h = (img_size[0] - h % img_size[0]) % img_size[0]
        pad_w = (img_size[1] - w % img_size[1]) % img_size[1]

        padded_image = cv2.copyMakeBorder(image_array, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Create patches
        img_patches = []
        for i in range(0, padded_image.shape[0], img_size[0]):
            for j in range(0, padded_image.shape[1], img_size[1]):
                patch = padded_image[i:i + img_size[0], j:j + img_size[1]] / 255.0  # Normalize
                patch = np.expand_dims(patch, axis=-1)  # Add channel dimension (1 for grayscale)
                img_patches.append(patch)

        return img_patches, (h, w), (padded_image.shape[0], padded_image.shape[1])


if __name__ == "__main__":
    root = tk.Tk()
    app = DnoOka(root)
    root.mainloop()
