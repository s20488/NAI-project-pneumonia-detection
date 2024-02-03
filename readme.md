# Detekcja Zapalenia Płuc na Zdjęciach Rentgenowskich

## Przegląd
Projekt ten został stworzony w celu wykrywania zapalenia płuc na zdjęciach rentgenowskich przy użyciu trzech różnych modeli wstępnie nauczonych: VGG19, ResNet50 i MobileNet. Modele te zostały trenowane na zestawie danych zawierającym oznaczone obrazy dla przypadków normalnych oraz zapalenia płuc.

## Struktura Projektu
Projekt jest zorganizowany w następujący sposób:

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
