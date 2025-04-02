# Pneumonia Detection in X-ray Images

This project is designed to detect pneumonia in chest X-ray images using three pre-trained deep learning models: VGG19, ResNet50, and MobileNet. The models were trained on a labeled dataset containing both normal and pneumonia cases.

## Getting Started
To set up the project, follow these steps:

### 1. Clone the Repository
Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/s20488/NAI_project_pneumonia_detection.git
```

### 2. Install Dependencies
Navigate to the project directory and install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset was sourced from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

Ensure the dataset is organized in the following structure:

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

### 4. Run the Flask Web Application
Launch the app and visit http://localhost:5000/ in your browser to upload and predict X-ray images.

```bash
python main.py
```
## Additional Notes
Portions of the code were adapted from this Kaggle notebook:
https://www.kaggle.com/code/karan842/pneumonia-detection-transfer-learning-94-acc/notebook#Importing-necessary-libraries.
