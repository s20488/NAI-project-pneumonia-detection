# Detekcja zapalenia płuc na zdjęciach X-ray

Projekt ten został stworzony w celu wykrywania zapalenia płuc na zdjęciach rentgenowskich przy użyciu trzech różnych modeli wstępnie nauczonych: VGG19, ResNet50 i MobileNet. Modele te zostały trenowane na zestawie danych zawierającym oznaczone obrazy dla przypadków normalnych oraz zapalenia płuc.

## Struktura projektu

1. **main.py**: Główny skrypt inicjujący aplikację internetową Flask do przesyłania i przewidywania obrazów rentgenowskich. Zawiera również funkcje do tworzenia diagramów i generowania generatorów danych obrazowych.

2. **model_vgg19.py**: Skrypt zawierający funkcję do trenowania modelu VGG19 do detekcji zapalenia płuc.

3. **model_resnet50.py**: Skrypt zawierający funkcję do trenowania modelu ResNet50 do detekcji zapalenia płuc.

4. **model_mobilenet.py**: Skrypt zawierający funkcję do trenowania modelu MobileNet do detekcji zapalenia płuc.
   
5. **test_callback.py**: Zawiera klasę do oceny metryk testowych na końcu każdej epoki treningowej.

6. **metrics.py**: Skrypt zawierający funkcje do generowania wykresów metryk precision, recall i F1 modeli na zbiorze testowym.

## Rozpoczęcie pracy
Aby uruchomić projekt, wykonaj następujące kroki:

### 1. Sklonuj repozytorium
Sklonuj to repozytorium na swój lokalny komputer za pomocą następującej komendy:

```bash
git clone https://github.com/twoja-nazwa-uzytkownika/detekcja-zapalenia-pluc.git
```

### 2. Zainstaluj zależności
Przejdź do katalogu projektu i zainstaluj wymagane biblioteki Pythona:

```bash
pip install -r requirements.txt
```

### 3. Pobierz zbiór danych
Dane zostały pobrane ze strony Kaggle za pomocą poniższego linku: 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

Upewnij się, że zbiór danych jest zorganizowany w następujący sposób:

```bash
data/
|-- chest_xray/
|   |-- train/
|       |-- NORMAL/
|       |-- PNEUMONIA/
|   |-- val/
|       |-- NORMAL/
|       |-- PNEUMONIA/
|   |-- test/
|       |-- NORMAL/
|       |-- PNEUMONIA/
```

### 4. Uruchom aplikację internetową Flask
Odwiedź http://localhost:5000/ w przeglądarce internetowej, aby przesyłać i przewidywać obrazy rentgenowskie.

```bash
python main.py
```
## Dodatkowa informacja
Niektóre fragmenty kodu zostały wykorzystane z poniższego linku: 
https://www.kaggle.com/code/karan842/pneumonia-detection-transfer-learning-94-acc/notebook#Importing-necessary-libraries.
