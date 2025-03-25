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
## Projekt implementuje 3 metody przetwarzania obrazów:
1. **Filtr Frangi'ego**
2. **Klasyfikator random forest**
3. **Model Unet**

# Filtr Frangiego
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
# Wizualizacja wyników działania programu
## Obraz 1
| ![Obraz 1](https://github.com/user-attachments/assets/91ede0b3-a25f-4cc3-82a1-398c7876ec53) | ![Obraz 2](https://github.com/user-attachments/assets/019638af-087a-4c4f-9ae9-63d88a0e074f) | ![11_rf](https://github.com/user-attachments/assets/5f8f0d45-c4d2-43b6-97d3-1f61f46e129d) | ![11_unet](https://github.com/user-attachments/assets/39c60dc7-f1c6-4bb3-b87c-0d563f0bf17d) | ![11_h](https://github.com/user-attachments/assets/d32c6af9-181f-4fe2-a7c7-14195f322f3e) | 
|:--:|:--:|:--:|:--:|:--:|
| Zdjęcie dna oka | Filtr Frangiego | Random Forest | UNet | Ground Truth |

## Obraz 2
| ![12_h](https://github.com/user-attachments/assets/e90041d5-18c6-48a3-818f-1b7e57f55310) | ![12_frangi](https://github.com/user-attachments/assets/5982d528-5c00-490a-9573-160d6d7b8981) | ![12_rf](https://github.com/user-attachments/assets/35118db7-6f42-4eed-ade2-779f94b6064f) | ![12_unet](https://github.com/user-attachments/assets/61b190b4-c928-45b1-b2b9-5eb1f7117f5a) | ![12_h_gt](https://github.com/user-attachments/assets/83b9c397-8200-46b9-82e7-948c64fe7b96) | 
|:--:|:--:|:--:|:--:|:--:|
| Zdjęcie dna oka | Filtr Frangiego | Random Forest | UNet | Ground Truth |

## Obraz 3
| ![13_h](https://github.com/user-attachments/assets/910c28ec-a564-4485-abce-383e9256db49) | ![13_frangi](https://github.com/user-attachments/assets/4907251a-f5b0-4c8e-ae13-9d870d30ebe9) | ![13_rf](https://github.com/user-attachments/assets/a694ab5b-4a33-46da-ab7b-6557a1070f87) | ![13_unet](https://github.com/user-attachments/assets/4775f35f-4414-4eab-8a3e-17425e23cfa4) | ![13_h_gt](https://github.com/user-attachments/assets/0626945d-97d9-4795-87e1-031027e4c5ed) | 
|:--:|:--:|:--:|:--:|:--:|
| Zdjęcie dna oka | Filtr Frangiego | Random Forest | UNet | Ground Truth |

## Obraz 4
| ![14_h](https://github.com/user-attachments/assets/5b055e09-8b5a-4088-b9d5-c035a8e9315f) | ![14_frangi](https://github.com/user-attachments/assets/aac5c5a6-1ac8-41da-a078-23c3ec82cd59) | ![14_rf](https://github.com/user-attachments/assets/4760558a-fb32-4eae-b0a0-db834681f3bc) | ![14_unet](https://github.com/user-attachments/assets/c357f315-2ed5-4ccd-88db-56e86985a44f) | ![14_h_gt](https://github.com/user-attachments/assets/3ae351ad-3cee-416b-951c-b244df0a018b) | 
|:--:|:--:|:--:|:--:|:--:|
| Zdjęcie dna oka | Filtr Frangiego | Random Forest | UNet | Ground Truth |

## Obraz 5
| ![15_h](https://github.com/user-attachments/assets/de23a935-26f0-4e8b-88c1-ff3c4598140d) | ![15_frangi](https://github.com/user-attachments/assets/699f3ddf-06f6-4b8c-bd19-ff096a632ac2) | ![15_rf](https://github.com/user-attachments/assets/5882c0cf-b0d8-49a7-9035-217843affca3) | ![15_unet](https://github.com/user-attachments/assets/7025e8af-90a9-4ba4-8511-d90dd8c7b573) | ![15_h_gt](https://github.com/user-attachments/assets/a1028ae2-8eb9-4029-95d3-ca1df1b26f4b) | 
|:--:|:--:|:--:|:--:|:--:|
| Zdjęcie dna oka | Filtr Frangiego | Random Forest | UNet | Ground Truth |


# Analiza wyników działania programu
Legenda
- f: Filtr Frangiego
- rf: Random Forest
- unet: UNet
## Wyniki
#### Obraz 1
| Model | Confusion Matrix | Accuracy | Sensitivity | Specificity | Mean Arithmetic | Mean Geometric |
|-------|------------------|----------|-------------|-------------|------------------|-----------------|
| f     | [[7311956, 108632], [318131, 446625]] | 0.9479   | 0.5840      | 0.9854      | 0.7847           | 0.7586          |
| rf    | [[7183939, 236649], [510250, 254506]] | 0.9088   | 0.3328      | 0.9681      | 0.6505           | 0.5676          |
| unet  | [[7280079, 140509], [223175, 541581]] | 0.9556   | 0.7082      | 0.9811      | 0.8446           | 0.8335          |

#### Obraz 2
| Model | Confusion Matrix | Accuracy | Sensitivity | Specificity | Mean Arithmetic | Mean Geometric |
|-------|------------------|----------|-------------|-------------|------------------|-----------------|
| f     | [[7181412, 146221], [297366, 560345]] | 0.9458   | 0.6533      | 0.9800      | 0.8167           | 0.8002          |
| rf    | [[7107890, 219743], [536649, 321062]] | 0.9076   | 0.3743      | 0.9700      | 0.6722           | 0.6026          |
| unet  | [[7180297, 147336], [242340, 615371]] | 0.9524   | 0.7175      | 0.9799      | 0.8487           | 0.8385          |

#### Obraz 3
| Model | Confusion Matrix | Accuracy | Sensitivity | Specificity | Mean Arithmetic | Mean Geometric |
|-------|------------------|----------|-------------|-------------|------------------|-----------------|
| f     | [[7452584, 21351], [615358, 96051]] | 0.9222   | 0.1350      | 0.9971      | 0.5661           | 0.3669          |
| rf    | [[7199138, 274797], [482618, 228791]] | 0.9075   | 0.3216      | 0.9632      | 0.6424           | 0.5566          |
| unet  | [[7320696, 153239], [231990, 479419]] | 0.9529   | 0.6739      | 0.9795      | 0.8267           | 0.8125          |

#### Obraz 4
| Model | Confusion Matrix | Accuracy | Sensitivity | Specificity | Mean Arithmetic | Mean Geometric |
|-------|------------------|----------|-------------|-------------|------------------|-----------------|
| f     | [[7254613, 216677], [243366, 470688]] | 0.9438   | 0.6592      | 0.9710      | 0.8151           | 0.8000          |
| rf    | [[7142680, 328610], [494606, 219448]] | 0.8994   | 0.3073      | 0.9560      | 0.6317           | 0.5420          |
| unet  | [[7334199, 137091], [243140, 470914]] | 0.9535   | 0.6595      | 0.9817      | 0.8206           | 0.8046          |

#### Obraz 5
| Model | Confusion Matrix | Accuracy | Sensitivity | Specificity | Mean Arithmetic | Mean Geometric |
|-------|------------------|----------|-------------|-------------|------------------|-----------------|
| f     | - | 0.9331   | 0.2021      | 0.9953      | 0.5987           | 0.4485          |
| rf    | [[7414476, 129623], [459200, 182045]] | 0.9281   | 0.2839      | 0.9828      | 0.6334           | 0.5282          |
| unet  | [[7464987, 79112], [266979, 374266]] | 0.9577   | 0.5837      | 0.9895      | 0.7866           | 0.7600          |

## Wnioski

#### Podsumowanie wydajności:
1. **Dokładność (Accuracy)**:
   - UNet osiąga najwyższą dokładność (95.2-95.8%)
   - Filtr Frangiego ma dobrą dokładność (92.2-94.8%)
   - Random Forest nieco słabszy (89.9-92.8%)

2. **Czułość (Sensitivity)**:
   - UNet najlepiej wykrywa naczynia (58-72%)
   - Filtr Frangiego - wyniki zmienne (13-66%)
   - Random Forest ma problemy z wykrywaniem (28-37%)

3. **Swoistość (Specificity)**:
   - Wszystkie modele mają wysoką swoistość (>95%)
   - Filtr Frangiego osiąga prawie idealną swoistość (97-99.7%)
   - UNet niewiele gorszy (98-99%)

---

#### Wnioski kliniczne:
1. **UNet**:
   - Najlepszy kompromis między wykrywaniem naczyń a unikaniem fałszywych alarmów
   - Najbardziej stabilny na różnych zestawach danych
   - Zalecany do zastosowań wymagających precyzyjnej segmentacji

2. **Filtr Frangiego**:
   - Doskonały do wykluczania przypadków negatywnych (mało fałszywie pozytywnych)
   - Może przegapiać niektóre naczynia (niska czułość w niektórych przypadkach)
   - Przydatny w przesiewowych badaniach

3. **Random Forest**:
   - Najsłabsze wyniki w wykrywaniu naczyń
   - Problemy z niezbalansowanymi klasami
   - Wymaga optymalizacji dla tego zadania

---

















