# Crop Disease Detection System

## Overview
The Crop Disease Detection System is a machine learning application designed to identify diseases in crop plants based on images of their leaves. Utilizing a pre-trained deep learning model, the system provides users with information about the detected disease, including its cause, symptoms, treatment recommendations, and prevention measures.

## Features
- Upload an image of a plant leaf to detect diseases.
- Get detailed information about the detected disease.
- Treatment and prevention recommendations for various crop diseases.

## Installation

### Prerequisites
Make sure you have Python 3.7 or higher installed on your machine.

### Clone the Repository
```
  git clone https://github.com/yourusername/crop-disease-detection.git
```
### Create a virtual environment and activate it:
```
python -m venv env
source venv/bin/activate  # On Windows, use `env\Scripts\activate`
```
### Install Dependencies
You can install the required packages using pip. It is recommended to use a virtual environment.
```
pip install -r requirements.txt
``` 

## Usage

### Training the Model
To train the model, run the following command:
  ```
  python train.py
  ```
This will create a model based on the images in the `data/train` directory and save the best model and class indices in the `models` directory.

### Running the Application
After training the model, you can run the Streamlit application:
  ```
    streamlit run app.py
  ```


### Uploading an Image
1. Click on the "Choose an image" button to upload an image of a plant leaf.
2. Click the "Analyze Image" button to get predictions and disease information.

## Dependencies
- tensorflow==2.13.0
- streamlit==1.24.0
- numpy==1.23.5
- Pillow==9.5.0
- pandas==2.0.3

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
