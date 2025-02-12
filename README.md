Below is a complete README file you can copy and paste into your project's README.md:

---

```markdown
# Face Rotation Classifier

This project implements a face rotation classifier that can detect and classify the rotation angle of faces in images. The system supports data augmentation (both on-the-fly and preprocessed), model training, evaluation, inference, and even face detection with cropping. All key settings (directories, augmentation parameters, model details, etc.) are centralized in a single configuration file (`config.yaml`).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Training and Testing](#training-and-testing)
  - [Inference](#inference)
  - [Face Detection](#face-detection)
  - [Visualization](#visualization)
  - [Streamlit Web App](#streamlit-web-app)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Flexible Data Augmentation:**  
  Choose between on-the-fly augmentation and preprocessed augmentation.
- **Multiple Model Options:**  
  Supports models such as ResNet18, ResNet34, ResNet50, ResNet101, VGG16, MobileNetV2, Inception, ViT, AlexNet, and a basic custom model.
- **Centralized Configuration:**  
  All hyperparameters and file/directory paths are stored in `config.yaml`.
- **Face Detection:**  
  Uses MTCNN for detecting and cropping faces from raw images.
- **Web Interface:**  
  A simple Streamlit app to upload an image and see the predicted rotation.
- **Visualization:**  
  Visualize augmented images in a grid to inspect augmentation quality.

## Project Structure

```
├── config.yaml                # Central configuration file
├── data_loader.py             # Unified dataset and preprocessing functions
├── model.py                   # Model definitions and utility function to get a model
├── preprocess.py              # Script for preprocessing images (augmentations saved to disk)
├── train.py                   # Training loop and validation logic
├── test.py                    # Testing script for evaluation on a test set
├── inference.py               # Inference script for a single image prediction
├── visualize.py               # Script to visualize augmented images
├── face_detect.py             # Script for face detection and cropping using MTCNN
├── streamlit_app.py           # Streamlit web interface for model inference
└── main.py                    # Main entry point to run preprocessing, training/testing, or inference
```

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/face-rotation-classifier.git
   cd face-rotation-classifier
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**

   All required packages are listed in `requirements.txt`. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If you plan to use CUDA, ensure you have a compatible PyTorch version installed.

## Configuration

All settings are managed in the `config.yaml` file. This file includes parameters for:
- **Augmentation:** Blur radius, color jitter, crop scale, noise intensity.
- **Model:** Model name (e.g., "resnet18"), number of classes, batch size, learning rate, epochs, and image size.
- **System:** Number of workers, CUDA usage, and process type (preprocessing, training+testing, or testing only).
- **Data Directories:** Paths for raw data, preprocessed data, training, and testing.
- **Face Detection:** Input and output directories and the maximum number of images to process.
  
Modify `config.yaml` as needed to suit your environment and requirements.

## Usage

### Preprocessing

To generate augmented images and organize them into class-specific folders:

1. Set `process_type` to `1` in `config.yaml`.
2. Run the main script:

   ```bash
   python main.py --config config.yaml
   ```

### Training and Testing

To train the model and then evaluate it:

1. Set `process_type` to `2` in `config.yaml`.
2. Run the main script:

   ```bash
   python main.py --config config.yaml
   ```

Training will save the best model (based on validation accuracy) into a designated folder under `saved_models/`.

### Inference

To predict the rotation angle of a single image:

1. Ensure that you have a trained model and update the model path if necessary.
2. Run the inference script:

   ```bash
   python inference.py test_image.jpg
   ```

This will display the image with its predicted rotation (e.g., 0°, 90°, etc.).

### Face Detection

To run face detection and crop faces from images:

1. Configure the `face_detect` section in `config.yaml` (set input directory, output directory, and optionally `max_images`).
2. Run the face detection script:

   ```bash
   python face_detect.py
   ```

Cropped face images will be saved in the specified output directory.

### Visualization

To visualize a grid of augmented images (default is a 5x5 grid):

```bash
python -c "from visualize import visualize_augmentations; visualize_augmentations(__import__('yaml').safe_load(open('config.yaml')))"
```

Or include a similar command in a separate script.

### Streamlit Web App

To run the web interface for image prediction:

1. Start the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Open the provided URL in your browser, upload an image, and view the predicted rotation.

## Troubleshooting

- **Palette Image Warnings / Conversion Issues:**  
  If you see warnings regarding palette images with transparency, the dataset loader attempts to convert these images properly. In cases of prolonged delays or errors, verify that the images in your dataset are not corrupted. The loader now uses a try/except block to handle errors gracefully.

- **Best Model Not Found:**  
  If training was stopped early and the experiment info is missing, the testing script will try to locate the best available model in the experiment folder. Adjust the `load_best_model_path` function in `test.py` if needed.

- **CUDA Issues:**  
  Ensure you have a compatible version of PyTorch and that CUDA is available if you intend to use GPU acceleration.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

