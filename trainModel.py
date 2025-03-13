import cv2
import numpy as np
import glob
import os
from skimage.feature import hog
from skimage.measure import moments_central, moments_hu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from sklearn.decomposition import PCA
def global_contrast_normalization(image):
    """Normalizacja globalnego kontrastu obrazu"""
    # Obliczenie średniej i odchylenia standardowego
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

def zero_phase_component_analysis(image):
    """Zero-Phase Component Analysis (ZPCA)"""
    # ZPCA wymaga uśredniania i centrowania obrazu
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    centered_image = image - mean

    # Wykorzystanie PCA do analizy składników obrazu
    pca = PCA(n_components=image.shape[2])
    pca.fit(centered_image.reshape(-1, image.shape[2]))
    transformed_image = pca.transform(centered_image.reshape(-1, image.shape[2]))
    zpca_image = pca.inverse_transform(transformed_image).reshape(image.shape)

    return zpca_image

def extract_features(image):
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
    #hog_features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)

    # Połączenie cech w jeden wektor
    features = [variance_r, variance_g, variance_b, central_moment] + list(hu_moments)


    return features

X = []  # cechy
y = []  # etykiety

image_files = sorted(glob.glob("healthy/*.jpg"))  # Ścieżka do obrazów
mask_files = sorted(glob.glob("healthy_vains/*.tif"))  # Ścieżka do masek eksperckich

for img_path, mask_path in zip(image_files, mask_files):
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    num_samples = 5000  # Maksymalna liczba próbek na obraz
    sampled_points = np.random.randint(4, image.shape[0] - 4, size=(num_samples, 2))

    for i, j in sampled_points:
        label = mask[i, j]  # Środkowy piksel maski jako etykieta
        feature = extract_features(image[i - 4:i + 4, j - 4:j + 4])  # Wycinamy okno 8x8
        X.append(feature)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening klasyfikatora (las losowy)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# Predykcja i ocena modelu
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Dokładność modelu: {accuracy:.4f}")
import joblib

# Zapisanie modelu do pliku
joblib.dump(clf, ".venv/random_forest_model.pkl")

print("Model zapisany jako 'random_forest_model.pkl'")
