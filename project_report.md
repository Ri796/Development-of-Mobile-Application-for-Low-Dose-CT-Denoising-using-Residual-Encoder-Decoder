# Project Report: RedCNN Medical Image Denoising

## 1. Project Overview
This project is a **Medical Image Denoising System** designed to reduce noise in Computed Tomography (CT) scans, specifically targeting low-dose CT images. It utilizes a Deep Learning model (**RED-CNN**) to enhance image quality, aiding in better diagnosis and analysis.

The system consists of a **Client-Server Architecture**:
-   **Mobile App (Client)**: A Flutter-based Android application for user interaction (uploading images, viewing results).
-   **Backend (Server)**: A Python FastAPI server that hosts the Deep Learning model and performs image processing.

---

## 2. Technical Architecture

### 2.1 Technology Stack
*   **Frontend**: Flutter (Dart)
*   **Backend**: Python, FastAPI, Uvicorn
*   **Deep Learning Framework**: PyTorch
*   **Image Processing**: Pydicom (for DICOM), Pillow, NumPy
*   **Networking**: REST API (HTTP POST)

### 2.2 Workflow
1.  **User Action**: User selects a DICOM (`.dcm`, `.ima`) or standard image (`.png`, `.jpg`) from the mobile app.
2.  **Upload**: The app sends the raw file to the backend via an HTTP POST request to `http://<server-ip>:8000/predict`.
3.  **Preprocessing (Server)**:
    *   **DICOM**: Reads pixel data, rescales to Hounsfield Units (HU) using Slope/Intercept, clips to Abdomen specific range `[-1024, 3072]`, and normalizes to `[0, 1]`.
    *   **Standard Image**: Converts to Grayscale, resizes to `256x256`, and normalizes.
4.  **Inference**: The processed Tensor is passed through the **RED-CNN** (Residual Encoder-Decoder Convolutional Neural Network) model.
5.  **Post-processing**:
    *   **Windowing**: Applies an "Abdomen Window" (Window Level: 50, Window Width: 240) to highlight soft tissues.
    *   **Conversion**: Converts both the original input and the denoised output into Base64-encoded PNG strings.
6.  **Display**: The server responds with the Base64 strings. The App decodes them and displays the **Input** vs **Denoised Output** side-by-side.
7.  **Save**: User can save the denoised result to the device gallery.

---

## 3. Component Analysis

### 3.1 Backend (`/backend`)
*   **`main.py`**: The core entry point.
    *   Initializes the `FastAPI` app.
    *   Loads the PyTorch model (`RED_CNN`) and weights (`REDCNN_100000iter.ckpt`).
    *   Defines the `/predict` endpoint to handle file uploads, run inference, and return results.
    *   Handles image resizing (256x256) for consistency.
*   **`prep.py`**: Handles specialized Medical Image logic.
    *   **rescale_slope/intercept**: Converts raw DICOM integers to physical Hounsfield Units.
    *   **apply_abdomen_window**: Transforms the high-dynamic-range CT data into a viewable range for human eyes (emphasizing organs over bone/air).
*   **`networks.py`**: Defines the Neural Network Architecture.
    *   **RED-CNN**: Contains 5 Convolutional layers (Encoder) and 5 Transpose Convolutional layers (Decoder) with Skip Connections (Residuals) to preserve details.

### 3.2 Frontend (`/redcnn_app`)
*   **`lib/main.dart`**: The complete UI and logic.
    *   **`FilePicker`**: Allows selecting files from the device storage.
    *   **`http`**: Manages the network request to the backend. Note: It is currently configured to point to a specific IP (`10.123.158.142`) which resolves to the host machine during development.
    *   **`Image.memory`**: Displays the Base64 decoded images.
    *   **`Gal`**: Plugin used to save the result image to the Photo Gallery.

---

## 4. Current Status
*   **Server**: Running on `0.0.0.0:8000` (accessible via local network).
*   **Model**: Loaded successfully from checkpoint.
*   **App Features**:
    *   File Selection (working for DICOM & Images).
    *   Communication with Server (working).
    *   Display of Input (Correctly windowed/processed) and Output.
    *   Save to Gallery functionality.

## 5. Potential Improvements
*   **Dynamic Resizing**: Currently fixed to 256x256. Could support full-resolution inference using tiling/patching.
*   **Windowing Controls**: Allow users to adjust Window Level/Width (e.g., for Lung or Bone windows) in the app.
*   **Batch Processing**: Support uploading multiple slices or a full scan volume.
