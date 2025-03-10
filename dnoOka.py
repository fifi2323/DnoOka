import tkinter as tk
from tkinter import ttk, filedialog, Label, Button, Canvas, Checkbutton, IntVar, DoubleVar, StringVar, Entry, simpledialog
from PIL import Image, ImageTk
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import time
import joblib
from skimage.feature import hog
from skimage.measure import moments_central, moments_hu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2


class DnoOka:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulator Tomografu Komputerowego")

        # Wczytanie modelu
        self.clf = joblib.load(".venv/random_forest_model.pkl")

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
        self.pred_button = Button(self.control_frame, text="Znajdź naczynia", command=self.predict)
        self.pred_button.grid(row=0, column=1, padx=5, pady=5)

        # Canvas do wyświetlania oryginalnego obrazu
        self.original_picture_canvas = Canvas(self.image_frame, width=400, height=400, bg="black")
        self.original_picture_canvas.grid(row=0, column=0, padx=15, pady=10)
        Label(self.image_frame, text="Oryginalny obraz").grid(row=1, column=0)

        # Canvas do wyświetlania obrazu po predykcji
        self.prediction_picture_canvas = Canvas(self.image_frame_pred, width=400, height=400, bg="black")
        self.prediction_picture_canvas.grid(row=0, column=0, padx=15, pady=10)
        Label(self.image_frame_pred, text="Obraz po predykcji").grid(row=1, column=0)

        self.image = None
        self.image_array = None
        self.reconstructed_image = None

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
        for i in range(4, self.image_array.shape[0] - 4, window_size):
            for j in range(4, self.image_array.shape[1] - 4, window_size):
                feature = self.extract_features(self.image_array[i - 4:i + 4, j - 4:j + 4])  # 8x8 okno
                features.append(feature)

        features = np.array(features)

        # Predykcja
        predictions = self.clf.predict(features)

        # Zbudowanie obrazu z predykcjami
        prediction_image = np.zeros(self.image_array.shape)
        index = 0
        for i in range(4, self.image_array.shape[0] - 4, window_size):
            for j in range(4, self.image_array.shape[1] - 4, window_size):
                prediction_image[i:i + window_size, j:j + window_size] = predictions[index]  # Przydziel predykcję
                index += 1
        print(prediction_image)
        # Konwertowanie obrazu predykcji do formatu wyświetlania
        predicted_image_pil = Image.fromarray(np.uint8(prediction_image * 255))  # Zamiana na 0-255
        self.predicted_image = ImageTk.PhotoImage(predicted_image_pil)
        # Wyświetlanie obrazu po predykcji
        self.prediction_picture_canvas.delete("all")
        self.prediction_picture_canvas.create_image(200, 200, image=self.predicted_image, anchor=tk.CENTER)
        print("finished")
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp;*.dcm")])
        if not file_path:
            return
        from PIL import Image
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
