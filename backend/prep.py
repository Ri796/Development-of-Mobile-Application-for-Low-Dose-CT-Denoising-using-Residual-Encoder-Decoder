import pydicom
import numpy as np
import torch

HU_MIN = -1024
HU_MAX = 3072

def load_and_preprocess_dicom(path):
    ds = pydicom.dcmread(path)

    img = ds.pixel_array.astype(np.float32)
    img = img * ds.RescaleSlope + ds.RescaleIntercept  # HU conversion

    # Fixed normalization for Model (matches training)
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)           # normalize to [0,1]

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    return img.float()



def apply_window(img_norm, wc, ww):
    # 1. De-normalize from [0, 1] back to HU
    img_hu = img_norm * (HU_MAX - HU_MIN) + HU_MIN
    
    # 2. Apply Window
    img_min = wc - ww / 2
    img_max = wc + ww / 2
    
    img_windowed = np.clip(img_hu, img_min, img_max)
    img_windowed = (img_windowed - img_min) / (img_max - img_min)
    
    return img_windowed

def apply_abdomen_window(img_norm):
    """
    Applies standard Abdomen/Soft Tissue window (WL: -160, WW: 240).
    Expects img_norm to be in strict HU_MIN/HU_MAX range [0, 1].
    """
    return apply_window(img_norm, wc=40, ww=400)
