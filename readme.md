# Detekcja Zapalenia Płuc na Zdjęciach Rentgenowskich

Projekt ten został stworzony w celu wykrywania zapalenia płuc na zdjęciach rentgenowskich przy użyciu trzech różnych modeli wstępnie nauczonych: VGG19, ResNet50 i MobileNet. Modele te zostały trenowane na zestawie danych zawierającym oznaczone obrazy dla przypadków normalnych oraz zapalenia płuc.

## Struktura Projektu
Projekt jest zorganizowany w następujący sposób:

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

1. **main.py**: Główny skrypt inicjujący aplikację internetową Flask do przesyłania i przewidywania obrazów rentgenowskich. Zawiera również funkcje do tworzenia diagramów, generowania generatorów danych obrazowych i trenowania modeli.

2. **model_vgg19.py**: Skrypt zawierający funkcję do trenowania modelu VGG19 do detekcji zapalenia płuc.

3. **model_resnet50.py**: Skrypt zawierający funkcję do trenowania modelu ResNet50 do detekcji zapalenia płuc.

4. **model_mobilenet.py**: Skrypt zawierający funkcję do trenowania modelu MobileNet do detekcji zapalenia płuc.

5. **metrics.py**: Zawiera niestandardowe metryki (F1 Score) i funkcje do generowania wykresów metryk ewaluacyjnych.

## Rozpoczęcie Pracy
Aby uruchomić projekt, wykonaj następujące kroki:

### 1. Sklonuj Repozytorium
Sklonuj to repozytorium na swój lokalny komputer za pomocą następującej komendy:

```bash
git clone https://github.com/twoja-nazwa-uzytkownika/detekcja-zapalenia-pluc.git
```

### 2. Zainstaluj Zależności
Przejdź do katalogu projektu i zainstaluj wymagane biblioteki Pythona:

```bash
pip install -r requirements.txt
```

### 3. Zbiór Danych
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
