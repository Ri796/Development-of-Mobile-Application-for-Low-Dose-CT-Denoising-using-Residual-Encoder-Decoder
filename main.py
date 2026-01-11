from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import torch
import numpy as np
import os
import io
from PIL import Image
import pydicom
from networks import RED_CNN
import prep

app = FastAPI()

# Setup Device and Model
DEVICE = "cpu"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "REDCNN_100000iter.ckpt")

model = RED_CNN().to(DEVICE)
if os.path.exists(CKPT_PATH):
    print(f"Loading model from {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
else:
    print(f"WARNING: Checkpoint not found at {CKPT_PATH}")

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess standard image (PNG/JPG)"""
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Grayscale
    image = image.resize((256, 256)) # Resize to expected input
    img_array = np.array(image).astype(np.float32)
    # Normalize 0-255 -> 0-1
    img_array = img_array / 255.0
    # Add dims: (1, 1, H, W)
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor

def preprocess_dicom(file_bytes: bytes) -> torch.Tensor:
    """Preprocess DICOM file"""
    # pydicom needs a file-like object
    dicom_file = io.BytesIO(file_bytes)
    ds = pydicom.dcmread(dicom_file)
    
    img = ds.pixel_array.astype(np.float32)
    
    # Simple resizing for DICOM if not 256x256?
    # The model expects fixed size maybe? REDCNN is fully conv, so it handles any size.
    # But usually patches/crops are used. 
    # For this demo, let's use the original size or resize if too huge.
    # Logic from prep.py:
    # img = img * ds.RescaleSlope + ds.RescaleIntercept
    # img = np.clip(img, HU_MIN, HU_MAX)
    # img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    
    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', -1024.0)
    
    img = img * slope + intercept
    
    HU_MIN = -1024
    HU_MAX = 3072
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    
    # RED-CNN typically trained on patches, but full image inference works if memory allows.
    # Let's resize only if requested or just keep as is?
    # To be safe and consistent with previous app logic, let's resize to 256x256 if possible, 
    # OR just return as is (model is FCN).
    # Let's resize to 256x256 for speed and consistency with prior Flutter code.
    
    # Resize using PIL
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((256, 256))
    img = np.array(img_pil).astype(np.float32) / 255.0

    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img_tensor

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith('.dcm') or filename.endswith('.ima'):
            input_tensor = preprocess_dicom(contents)
        else:
            input_tensor = preprocess_image(contents)
            
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        import base64
        
        # --- Helper for Tensor to Base64 PNG ---
        def tensor_to_base64(tensor):
            arr = tensor.squeeze().cpu().numpy()
            # Apply Windowing (Important for DICOM)
            arr = prep.apply_abdomen_window(arr)
            arr = np.clip(arr, 0, 1)
            pil_img = Image.fromarray((arr * 255).astype(np.uint8))
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            return base64.b64encode(buff.getvalue()).decode('utf-8')

        # Generate Base64 Strings
        input_b64 = tensor_to_base64(input_tensor)
        output_b64 = tensor_to_base64(output_tensor)
        
        # Input is 'input_tensor' (1,1,H,W), Output is 'output_tensor' (1,1,H,W)
        import metrics
        
        # Calculate Metrics (on normalized tensors [0,1])
        psnr_val = metrics.calculate_psnr(input_tensor, output_tensor)
        ssim_val = metrics.calculate_ssim(input_tensor, output_tensor)
        rmse_val = metrics.calculate_rmse(input_tensor, output_tensor)
        
        return {
            "input_preview": input_b64,
            "output_image": output_b64,
            "metrics": {
                "psnr": round(psnr_val, 2),
                "ssim": round(ssim_val, 3),
                "rmse": round(rmse_val, 4)
            }
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "RedCNN Server Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
