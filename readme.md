# Detekcja zapalenia płuc na zdjęciach X-ray

Projekt ten został stworzony w celu wykrywania zapalenia płuc na zdjęciach rentgenowskich przy użyciu trzech różnych modeli wstępnie nauczonych: VGG19, ResNet50 i MobileNet. Modele te zostały trenowane na zestawie danych zawierającym oznaczone obrazy dla przypadków normalnych oraz zapalenia płuc.

## Rozpoczęcie pracy
Aby uruchomić projekt, wykonaj następujące kroki:

### 1. Sklonuj repozytorium
Sklonuj to repozytorium na swój lokalny komputer za pomocą następującej komendy:

```bash
git clone https://github.com/s20488/NAI_project_pneumonia_detection.git
```

### 2. Zainstaluj zależności
Przejdź do katalogu projektu i zainstaluj wymagane biblioteki Pythona:

```bash
pip install -r requirements.txt
```

### 3. Pobierz zbiór danych
Dane zostały pobrane ze strony Kaggle: 
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
