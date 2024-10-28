# Crop Disease Detection System

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Data Preprocessing](#data-preprocessing)
- [Disease Information](#disease-information)
- [License](#license)

## Description

The Crop Disease Detection System is a machine learning application designed to identify diseases in crop plants based on images of their leaves. By leveraging deep learning techniques, this system provides accurate predictions and treatment recommendations for various plant diseases.

## Features

- **Image Upload**: Users can upload images of plant leaves for analysis.
- **Disease Prediction**: The system predicts the disease affecting the plant with a confidence score.
- **Treatment Recommendations**: Provides detailed treatment and prevention measures for detected diseases.
- **User-Friendly Interface**: Built using Streamlit for an interactive user experience.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```
2. Create python environment and activate it :

   ```
   python -m venv environment_name
   ```

   ```
   environment_name\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

## Model Training

The model is built using TensorFlow and Keras. To train the model, run the following command:

```
python train.py
```

To run the application, execute the following command:

```
streamlit run app.py
```

### Uploading an Image

- Navigate to the web interface.
- Use the file uploader to select an image of a plant leaf.
- Click on "Analyze Image" to get predictions.

## Data Preprocessing

The data preprocessing is handled and It includes functions for:

- Resizing images
- Normalizing pixel values
- Creating data generators for training and validation

## Disease Information

The `utils/disease_info.py` file contains detailed information about various plant diseases, including:

- **Cause**: The pathogen responsible for the disease.
- **Symptoms**: Visual indicators of the disease.
- **Treatment**: Recommended actions to treat the disease.
- **Prevention**: Measures to prevent the disease from occurring.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Special Note

Due to Large dataset the dataset is not provided but here is kaggle dataset link:
https://www.kaggle.com/datasets/emmarex/plantdisease

### Steps:

- Download the Dataset
- Split it into train(0.8) and test(0.2)
- Save into folder named data use it for model training
