from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
from typing import Dict, Any
import uvicorn

app = FastAPI(title="Image Processing API", version="1.0.0")

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image processing functions from your notebook
def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    
    return output.astype(np.uint8)

def gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_blur(image, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return convolve(image, kernel)

def laplacian_sharpen(image):
    laplacian_kernel = np.array([
        [0, -1,  0],
        [-1,  4, -1],
        [0, -1,  0]
    ])
    laplacian = convolve(image, laplacian_kernel)
    sharpened = np.clip(image + laplacian, 0, 255)
    return sharpened.astype(np.uint8)

def local_variance(image, ksize=5):
    mean = cv2.blur(image.astype(np.float32), (ksize, ksize))
    mean_sq = cv2.blur(np.square(image.astype(np.float32)), (ksize, ksize))
    variance = mean_sq - np.square(mean)
    return variance

def adaptive_blur_sharpen_color(image, blur_sigma=1.0, ksize=5, var_threshold=500.0):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a color image (H x W x 3)")
    
    channels = cv2.split(image)
    result_channels = []
    for c in channels:
        blurred = gaussian_blur(c, size=ksize, sigma=blur_sigma)
        sharpened = laplacian_sharpen(c)
        var_map = local_variance(c, ksize)
        combined = np.where(var_map < var_threshold, blurred, sharpened)
        result_channels.append(combined.astype(np.uint8))
    return cv2.merge(result_channels)

def improved_adaptive_filter(image, blur_sigma=1.0, ksize=5, var_threshold_low=50.0, var_threshold_high=1000.0):
    channels = cv2.split(image)
    result_channels = []
    
    for c in channels:
        c = c.astype(np.uint8)
        blurred = gaussian_blur(c, size=ksize, sigma=blur_sigma)
        sharpened = laplacian_sharpen(c)
        var_map = local_variance(c, ksize)
        
        # Normalize to [0, 1] based on thresholds
        alpha = np.clip((var_map - var_threshold_low) / (var_threshold_high - var_threshold_low), 0, 1)
        # Blend based on alpha: more blur in low-var, more sharpen in high-var
        output = (1 - alpha) * blurred + alpha * sharpened
        result_channels.append(np.clip(output, 0, 255).astype(np.uint8))
    
    return cv2.merge(result_channels)

def apply_per_channel(image, filter_func):
    channels = cv2.split(image)
    processed = [filter_func(c) for c in channels]
    return cv2.merge(processed)

def psnr(original, processed):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255**2 / mse)

def psnr_color(original, processed):
    channels_orig = cv2.split(original)
    channels_proc = cv2.split(processed)
    return np.mean([psnr(o, p) for o, p in zip(channels_orig, channels_proc)])

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

@app.get("/")
async def root():
    return {"message": "Image Processing API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is working properly"}

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    blur_sigma: float = 1.0,
    kernel_size: int = 5,
    var_threshold: float = 500.0,
    var_threshold_low: float = 50.0,
    var_threshold_high: float = 1000.0
):
    """
    Process uploaded image with various filters and return results with PSNR metrics
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Get image dimensions
        height, width, channels = original_image.shape
        
        # Apply different filters
        results = {}
        
        # 1. Original image
        original_b64 = image_to_base64(original_image)
        results["original"] = {
            "name": "Original",
            "description": "Original uploaded image",
            "image": original_b64,
            "psnr": float('inf'),
            "parameters": {}
        }
        
        # 2. Gaussian Blur
        blurred_color = apply_per_channel(original_image, 
                                        lambda img: gaussian_blur(img, size=kernel_size, sigma=blur_sigma))
        blur_psnr = psnr_color(original_image, blurred_color)
        results["gaussian_blur"] = {
            "name": "Gaussian Blur",
            "description": "Smoothing filter that reduces noise and detail",
            "image": image_to_base64(blurred_color),
            "psnr": round(blur_psnr, 2),
            "parameters": {
                "sigma": blur_sigma,
                "kernel_size": kernel_size
            }
        }
        
        # 3. Laplacian Sharpening
        sharpened_color = apply_per_channel(original_image, laplacian_sharpen)
        sharpen_psnr = psnr_color(original_image, sharpened_color)
        results["laplacian_sharpen"] = {
            "name": "Laplacian Sharpening",
            "description": "Edge enhancement filter that increases detail and contrast",
            "image": image_to_base64(sharpened_color),
            "psnr": round(sharpen_psnr, 2),
            "parameters": {
                "kernel": "Laplacian 3x3"
            }
        }
        
        # 4. Adaptive Filter (Original)
        adaptive_color = adaptive_blur_sharpen_color(
            original_image, 
            blur_sigma=blur_sigma, 
            ksize=kernel_size, 
            var_threshold=var_threshold
        )
        adaptive_psnr = psnr_color(original_image, adaptive_color)
        results["adaptive_filter"] = {
            "name": "Adaptive Filter",
            "description": "Applies blur to smooth areas and sharpening to detailed areas",
            "image": image_to_base64(adaptive_color),
            "psnr": round(adaptive_psnr, 2),
            "parameters": {
                "blur_sigma": blur_sigma,
                "kernel_size": kernel_size,
                "variance_threshold": var_threshold
            }
        }
        
        # 5. Improved Adaptive Filter
        improved_adaptive = improved_adaptive_filter(
            original_image,
            blur_sigma=blur_sigma,
            ksize=kernel_size,
            var_threshold_low=var_threshold_low,
            var_threshold_high=var_threshold_high
        )
        improved_psnr = psnr_color(original_image, improved_adaptive)
        results["improved_adaptive"] = {
            "name": "Improved Adaptive Filter",
            "description": "Advanced adaptive filter with smooth blending between blur and sharpen",
            "image": image_to_base64(improved_adaptive),
            "psnr": round(improved_psnr, 2),
            "parameters": {
                "blur_sigma": blur_sigma,
                "kernel_size": kernel_size,
                "var_threshold_low": var_threshold_low,
                "var_threshold_high": var_threshold_high
            }
        }
        
        # Summary statistics
        summary = {
            "image_info": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "filename": file.filename
            },
            "processing_results": {
                "total_filters": len(results) - 1,  # Exclude original
                "best_psnr": {
                    "filter": max(
                        [(k, v["psnr"]) for k, v in results.items() if v["psnr"] != float('inf')],
                        key=lambda x: x[1]
                    )[0],
                    "value": max([v["psnr"] for v in results.values() if v["psnr"] != float('inf')])
                }
            }
        }
        
        return {
            "success": True,
            "summary": summary,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-single-filter")
async def process_single_filter(
    file: UploadFile = File(...),
    filter_type: str = "gaussian_blur",
    blur_sigma: float = 1.0,
    kernel_size: int = 5,
    var_threshold: float = 500.0,
    var_threshold_low: float = 50.0,
    var_threshold_high: float = 1000.0
):
    """
    Process image with a single specified filter
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Apply selected filter
        if filter_type == "gaussian_blur":
            processed = apply_per_channel(original_image, 
                                        lambda img: gaussian_blur(img, size=kernel_size, sigma=blur_sigma))
        elif filter_type == "laplacian_sharpen":
            processed = apply_per_channel(original_image, laplacian_sharpen)
        elif filter_type == "adaptive_filter":
            processed = adaptive_blur_sharpen_color(
                original_image, blur_sigma=blur_sigma, ksize=kernel_size, var_threshold=var_threshold
            )
        elif filter_type == "improved_adaptive":
            processed = improved_adaptive_filter(
                original_image, blur_sigma=blur_sigma, ksize=kernel_size,
                var_threshold_low=var_threshold_low, var_threshold_high=var_threshold_high
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid filter type")
        
        psnr_value = psnr_color(original_image, processed)
        
        return {
            "success": True,
            "filter_type": filter_type,
            "original_image": image_to_base64(original_image),
            "processed_image": image_to_base64(processed),
            "psnr": round(psnr_value, 2),
            "image_info": {
                "width": original_image.shape[1],
                "height": original_image.shape[0],
                "channels": original_image.shape[2]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/available-filters")
async def get_available_filters():
    """
    Get list of available filters with their descriptions
    """
    filters = {
        "gaussian_blur": {
            "name": "Gaussian Blur",
            "description": "Smoothing filter that reduces noise and detail",
            "parameters": ["blur_sigma", "kernel_size"]
        },
        "laplacian_sharpen": {
            "name": "Laplacian Sharpening", 
            "description": "Edge enhancement filter that increases detail and contrast",
            "parameters": []
        },
        "adaptive_filter": {
            "name": "Adaptive Filter",
            "description": "Applies blur to smooth areas and sharpening to detailed areas",
            "parameters": ["blur_sigma", "kernel_size", "var_threshold"]
        },
        "improved_adaptive": {
            "name": "Improved Adaptive Filter",
            "description": "Advanced adaptive filter with smooth blending between blur and sharpen",
            "parameters": ["blur_sigma", "kernel_size", "var_threshold_low", "var_threshold_high"]
        }
    }
    
    return {
        "success": True,
        "filters": filters
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)