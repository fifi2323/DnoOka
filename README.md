# Skład grupy  
- **Filip Urbański** (155875)  
- **Paweł Kelar** (156023)  

## Technologia  
Projekt został stworzony za pomocą języka **Python 3.12**.  

## Dodatkowe biblioteki  
- `tkinter`  
- `PIL`  
- `numpy`  
- `tensorflow`  
- `glob`  
- `sklearn`  
- `cv2`  
- `joblib`  
- `matplotlib`  
# Przetwarzanie obrazów
Proces wykrywania naczyń krwionośnych na obrazie składa się z następujących etapów:
1. **Wstępne przetwarzanie obrazu**
2. **Zastosowanie filtru Frangi'ego do wykrycia naczyń**
3. **Postprocessing obrazu: progowanie i operacje morfologiczne**
4. **Wyświetlenie końcowego wyniku**

## Opis poszczególnych etapów

### 1. Wstępne przetwarzanie obrazu
Wstępne przetwarzanie obrazu obejmuje normalizację wartości pikseli do zakresu [0,1] w celu przygotowania obrazu do dalszego przetwarzania.

```python
normalized = preprocessed_image / 255.0
```

### 2. Zastosowanie filtru Frangi'ego
Filtr Frangi'ego jest stosowany do wykrywania naczyń krwionośnych poprzez analizę struktur przypominających rurki. Filtr działa poprzez analizę wartości własnych hessianu obrazu, co pozwala na wzmocnienie obszarów o wysokiej krzywiźnie.

```python
vessel_enhanced = frangi(normalized)
```

### 3. Postprocessing obrazu
Aby uzyskać binarną maskę naczyń:
- **Progowanie** obrazu po filtrze Frangi'ego, aby wyodrębnić struktury naczyń.
- **Operacje morfologiczne** (otwarcie i zamknięcie) w celu redukcji szumów i poprawy jakości wykrytej struktury naczyń.
- Usunięcie niepotrzebnych pikseli z obszarów, które w oryginalnym obrazie miały wartość zerową.

```python
binary_mask = (vessel_image > threshold).astype(np.uint8) * 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
```

Ostatecznie, jeśli obraz oryginalny zawierał czarne obszary, te same obszary są również zerowane w przetworzonym obrazie:

```python
if self.image_array is not None:
    if len(self.image_array.shape) == 3:
        original_gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = self.image_array
    closed[original_gray == 0] = 0
```

### 4. Wyświetlenie końcowego wyniku
Proces kończy się uruchomieniem pipeline'u przetwarzania obrazu, a następnie wyświetleniem wyniku w formacie obrazu PIL.

```python
preprocessed = self.preprocess_image(self.image_array)
vessel_enhanced = self.frangi_extract_vessels(preprocessed)
mask = self.frangi_postprocess_vessel_image(vessel_enhanced, threshold=0.001)
self.print_result(mask)

np_image_8bit = (mask * 255).astype(np.uint8)
self.predicted_image_pil = Image.fromarray(mask)
```

Dodatkowo, zmierzony jest czas przetwarzania obrazu:

```python
start = time.time()
...
stop = time.time()
print(f"Czas przetwarzania: {stop - start}")
```

## Uzasadnienie zastosowanego rozwiązania
- **Filtr Frangi'ego** został wybrany, ponieważ jest skuteczny w wykrywaniu struktur przypominających naczynia krwionośne, analizując ich orientację i krzywiznę.
- **Normalizacja obrazu** pozwala na lepsze działanie filtrów opartych na wartościach własnych hessianu.
- **Progowanie** pozwala na wyodrębnienie wykrytych naczyń.
- **Operacje morfologiczne** eliminują szumy i poprawiają jakość binarnej maski.
- **Eliminacja tła** zapewnia, że czarne obszary z oryginalnego obrazu nie zostaną błędnie zaklasyfikowane jako naczynia.

Dzięki zastosowaniu tych technik uzyskano skuteczne wykrywanie naczyń na obrazie, przy jednoczesnym zachowaniu wysokiej jakości wyników.

# Klasyfikator random forest

## Przygotowanie danych
### Wycinki 8x8
Do przygotowania danych, obraz jest dzielony na wycinki o rozmiarze 8x8 pikseli. Wycinki są losowo wybierane z obrazów z użyciem `np.random.randint`, generując 5000 próbek na obraz. Następnie dla każdego wycinka obliczane są cechy, które zostaną użyte do klasyfikacji.

```python
num_samples = 5000  # Maksymalna liczba próbek na obraz
sampled_points = np.random.randint(4, image.shape[0] - 4, size=(num_samples, 2))
```
### Extrakcja cech
Dla każdego wycinka obrazu ekstraktowane są różne cechy:
- **Wariancja kolorów:** Obliczana jest wariancja dla każdego z kanałów RGB.
- **Momenty centralne:** Służą do analizy rozkładu obrazu w przestrzeni.
- **Momenty Hu:** Zestaw siedmiu momentów, które są używane do analizy kształtu obrazu.
```python
def extract_features(image):
    variance_r = np.var(image[:, :, 0])
    variance_g = np.var(image[:, :, 1])
    variance_b = np.var(image[:, :, 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = moments_central(gray)
    if moments[0, 0] == 0:
        central_moment = 0
    else:
        central_moment = moments[0, 2] + moments[2, 0]
    hu_moments = moments_hu(moments)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    hu_moments = np.nan_to_num(hu_moments, nan=0.0, posinf=0.0, neginf=0.0)
    features = [variance_r, variance_g, variance_b, central_moment] + list(hu_moments)
    return features
```
## Wstępne przetwarzanie zbioru uczącego
Zbiór uczący jest podzielony na dane treningowe i testowe z użyciem train_test_split. Używa się 80% danych do treningu i 20% do testów. Do nauki wykorzystywane jest 10 obrazów z bazy HRF, pozostałe 5 będzie służyło do weryfikacji poprawności modelu. 
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Zastosowane metody uczenia maszynowego
Do klasyfikacji zastosowano algorytm lasu losowego, który jest zbiorem wielu drzew decyzyjnych. Jest to technika, która dobrze radzi sobie z klasyfikacją w przypadkach o dużych zbiorach danych i dużej liczbie cech.
- Liczba drzew (n_estimators): Ustawiona na 100.
- Losowość (random_state): Ustalono stałą wartość dla powtarzalności wyników.
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```
## Wyniki wstępnej oceny klasyfikatora (testy hold-out)
Po treningu modelu, przeprowadzono predykcję na zbiorze testowym i obliczono dokładność modelu za pomocą miary accuracy_score.
```python
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.4f}")
```
**Uzyskane wyniki na zbiorze walidacyjnym:**
- **Accuracy:** `0.93`

## Uzasadnienie zastosowanego rozwiązania
Las losowy został wybrany z powodu jego zdolności do radzenia sobie z dużą liczbą cech i prób, nawet w obecności szumów. Ponadto, jego interpretowalność oraz dobra wydajność w zadaniach klasyfikacji sprawiają, że jest to solidna opcja w przypadku klasyfikacji obrazów, szczególnie z dużymi i złożonymi danymi.
# Model UNet

## Przygotowanie danych

### 1. Wyznaczanie wycinków obrazu
W celu segmentacji naczyń krwionośnych obrazy są dzielone na mniejsze wycinki o stałym rozmiarze 512x512 pikseli. Proces ten obejmuje:
- Wczytanie obrazu oraz odpowiadającej mu maski naczyń.
- Normalizację wartości pikseli do przedziału [0,1].
- Podział obrazu na nie nakładające się fragmenty.
- Dopasowanie wymiarów do modelu (np. zmiana rozmiaru do 256x256).

```python
def preprocess_image(image_path, image_path_mask, img_size=(512, 512)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_mask = cv2.imread(image_path_mask, cv2.IMREAD_GRAYSCALE)

    h, w = img.shape
    pad_h = (img_size[0] - h % img_size[0]) % img_size[0]
    pad_w = (img_size[1] - w % img_size[1]) % img_size[1]

    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    img_mask = cv2.copyMakeBorder(img_mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    img_arr, img_mask_arr = [], []
    for i in range(0, img.shape[0] - img_size[0], img_size[0]):
        for j in range(0, img.shape[1] - img_size[1], img_size[1]):
            temp = img[i:i + img_size[0], j:j + img_size[1]] / 255.0
            temp_mask = img_mask[i:i + img_size[0], j:j + img_size[1]] / 255.0
            temp_mask = cv2.resize(temp_mask, (256, 256))
            img_arr.append(np.expand_dims(temp, axis=-1))
            img_mask_arr.append(np.expand_dims(temp_mask, axis=-1))
    
    return img_arr, img_mask_arr
```
---

## Wstępne przetwarzanie zbioru uczącego
- Dane obrazowe są dzielone na zbiór treningowy i walidacyjny w stosunku 80:20. Do nauki wykorzystywane jest 10 obrazów z bazy HRF, pozostałe 5 będzie służyło do weryfikacji poprawności modelu.
- Maski są przekształcane do postaci akceptowalnej przez model U-Net (rozmiar 256x256x1).
- Normalizacja wartości wejściowych pozwala na stabilniejsze trenowanie sieci neuronowej.

```python
train_images, val_images, train_masks, val_masks = train_test_split(
    train_images, train_masks, test_size=0.2, random_state=42
)

train_masks = train_masks.reshape(-1, 256, 256, 1)
val_masks = val_masks.reshape(-1, 256, 256, 1)
```

---

## Zastosowane metody uczenia maszynowego
Do segmentacji naczyń krwionośnych zastosowano **konwolucyjną sieć neuronową U-Net**. Jest to model oparty na architekturze enkoder-dekoder, dobrze sprawdzający się w zadaniach segmentacji medycznej.

**Przyjęte parametry:**
- **Struktura:** 5 poziomów, 64-256 filtrów na warstwę.
- **Aktywacja:** `relu` dla warstw konwolucyjnych, `sigmoid` dla wyjściowej.
- **Optymalizator:** `adam`
- **Funkcja straty:** `binary_crossentropy`
- **Metryka:** `accuracy`
- **Rozmiar batcha:** `8`
- **Liczba epok:** `30`

```python
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    output = Conv2D(1, 1, activation='sigmoid')(conv5)
    
    return Model(inputs, output)

model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## Wyniki wstępnej oceny klasyfikatora (testy hold-out)
Model został oceniony na zbiorze walidacyjnym przy użyciu klasycznego podziału na zbiór treningowy i testowy (*hold-out validation*).

**Uzyskane wyniki na zbiorze walidacyjnym:**
- **Loss:** `0.15`
- **Accuracy:** `0.94`

```python
val_loss, val_accuracy = model.evaluate(val_images, val_masks)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")
```



---

## Uzasadnienie zastosowanego rozwiązania
- **Podział obrazu na wycinki** pozwala na przetwarzanie dużych obrazów w mniejszych fragmentach, co poprawia skuteczność segmentacji.
- **Normalizacja i wstępne przetwarzanie** zapewniają lepszą stabilność podczas treningu modelu.
- **Sieć U-Net** jest standardem w segmentacji obrazów biomedycznych i umożliwia skuteczne odwzorowanie struktur naczyń.
- **Optymalizacja modelu** poprzez `EarlyStopping` i `ReduceLROnPlateau` zapobiega nadmiernemu dopasowaniu modelu i poprawia jakość segmentacji.

Całość procesu została zweryfikowana poprzez testy *hold-out*, a uzyskane wyniki potwierdzają skuteczność zastosowanego podejścia.
