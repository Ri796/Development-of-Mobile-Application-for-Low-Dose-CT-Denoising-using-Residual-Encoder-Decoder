# RedCNN Denoising App - Project Report

## 1. Project Overview
This project implements a **Low-Dose CT Image Denoising** system using a **RedCNN** (Residual Encoder-Decoder Convolutional Neural Network) model.

To ensure stability and performance on mobile devices, the system uses a **Client-Server Architecture**:
*   **Backend (Server):** Python-based API that runs the heavy AI model.
*   **Frontend (Client):** Flutter-based mobile app that acts as the user interface.

## 2. Directory Structure
The project is organized into two main directories:

```
final/
├── backend/                # The Python Server (The "Brain")
│   ├── main.py             # FastAPI entry point
│   ├── networks.py         # RedCNN model definition
│   ├── prep.py             # DICOM windowing & preprocessing
│   ├── requirements.txt    # Python dependencies
│   └── REDCNN_100000iter.ckpt # Trained model weights
│
├── redcnn_app/             # The Flutter Mobile App (The "Remote")
│   ├── lib/
│   │   └── main.py         # Main UI and logic (Dart)
│   ├── android/            # Android builds configs
│   └── pubspec.yaml        # App dependencies
│
└── temp_redcnn/            # Original source code (Archive)
```

## 3. Component Details

### A. Backend (`/backend`)
The backend is responsible for all image processing and inference.

*   **`main.py`**: The core server script.
    *   **Framework**: FastAPI.
    *   **Role**: Initializes the RedCNN model, listens on port `8000`, and exposes the `/predict` endpoint.
    *   **Logic**: Accepts an image/DICOM -> Preprocesses it -> Runs Inference -> Apples Windowing -> Returns JSON with Base64 images.
*   **`networks.py`**: Defines the `RED_CNN` PyTorch class.
    *   **Architectue**: A 10-layer convolutional network with residual skip connections.
*   **`prep.py`**: Utility functions for medical imaging.
    *   **Key Function**: `apply_abdomen_window`. Converts raw Hounsfield Units (HU) to a visible grayscale image optimized for soft tissue viewing.
*   **`REDCNN_100000iter.ckpt`**: The pre-trained PyTorch model file containing the learnable weights.

### B. Frontend (`/redcnn_app`)
The frontend provides the user experience on the mobile device.

*   **`lib/main.dart`**: The entire application logic.
    *   **Capabilities**:
        *   File Picking (Supports `.dcm`, `.ima`, `.png`, `.jpg`).
        *   HTTP Uploads (Sends data to the specific PC IP: `10.123.158.142`).
        *   Base64 Decoding (Converts server response strings back to images).
        *   Image Gallery Saving (Uses `gal` package).
*   **`pubspec.yaml`**: Configuration file listing packages:
    *   `http`: For network requests.
    *   `file_picker`: For selecting files.
    *   `gal`: For saving images to the gallery.

## 4. Work Flow
1.  **User Action**: User clicks "Pick Image/DICOM" on the mobile app.
2.  **Upload**: The app sends the file to the Backend Server (`/predict`).
3.  **Preprocessing (Server)**:
    *   If DICOM: Reads pixel data, applies rescale slope/intercept.
    *   If Image: Grayscale conversion and resizing to 256x256.
4.  **Inference (Server)**: The `RED_CNN` model processes the noisy input tensor and generates a denoised tensor.
5.  **Post-Processing (Server)**:
    *   **Windowing**: An Abdomen Window (Center: 50, Width: 240) is applied to both the input (for preview) and output.
    *   **Encoding**: Images are converted to Base64 strings.
6.  **Response**: The server returns a JSON object: `{"input_preview": "...", "output_image": "..."}`.
7.  **Display (App)**: The app decodes the strings and displays the Before/After comparison.
8.  **Save**: User clicks "Save Output" to store the result in the phone's gallery.

## 5. How to Run

### Step 1: Start the Backend
Ensure your PC is connected to WiFi.
Open a terminal in `final/backend/`:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*Note: This must remain running for the app to work.*

### Step 2: Run the App
**Option A: Development Mode (Requires USB)**
Open a terminal in `final/redcnn_app/`:
```bash
flutter run
```

**Option B: Standalone (APK)**
Install the generated APK (`build/app/outputs/flutter-apk/app-release.apk`) on your phone.
Ensure your phone is on the **same WiFi network** as the PC.
