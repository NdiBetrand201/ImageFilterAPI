
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import logging
from typing import Callable
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Processing API", version="1.0.0")

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image processing functions
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
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
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
        
        alpha = np.clip((var_map - var_threshold_low) / (var_threshold_high - var_threshold_low), 0, 1)
        output = (1 - alpha) * blurred + alpha * sharpened
        result_channels.append(np.clip(output, 0, 255).astype(np.uint8))
    
    return cv2.merge(result_channels)

def apply_per_channel(image, func: Callable) -> np.ndarray:
    channels = cv2.split(image)
    processed = [func(c) for c in channels]
    return cv2.merge(processed)

def psnr(original, processed):
    original = original.astype(np.float32)
    processed = processed.astype(np.float32)
    mse = np.mean((original - processed) ** 2)
    if mse == 0 or np.isnan(mse):
        return 100.0  # Use finite value instead of inf or NaN
    return 10 * np.log10(255**2 / mse)

def psnr_color(original: np.ndarray, processed: np.ndarray) -> float:
    channels_orig = cv2.split(original)
    channels_proc = cv2.split(processed)
    psnr_values = [psnr(o, p) for o, p in zip(channels_orig, channels_proc)]
    mean_psnr = np.mean([v for v in psnr_values if not np.isnan(v)])
    return 100.0 if np.isnan(mean_psnr) else round(mean_psnr, 2)

def image_to_base64(image: np.ndarray) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_string: str) -> np.ndarray:
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.get("/")
async def root():
    return {"message": "Image Processing API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is working properly"}

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    blur_sigma: float = Form(1.0),
    kernel_size: int = Form(5),
    var_threshold: float = Form(500.0),
    var_threshold_low: float = Form(50.0),
    var_threshold_high: float = Form(1000.0)
):
    try:
        if kernel_size % 2 == 0:
            raise HTTPException(status_code=400, detail="Kernel size must be an odd number")
        if var_threshold_high <= var_threshold_low:
            raise HTTPException(status_code=400, detail="High threshold must be greater than low threshold")
        
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 10MB limit")
        
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Please use JPG or PNG format")
        
        if not any(file.filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            raise HTTPException(status_code=400, detail="Only JPG and PNG formats are supported")
        
        logger.info(f"Processing image: {file.filename}, Size: {original_image.shape}")
        
        height, width, channels = original_image.shape
        results = {}
        original_b64 = image_to_base64(original_image)
        results["original"] = {
            "name": "Original",
            "description": "Original uploaded image",
            "image": original_b64,
            "psnr": 100.0,  # Use finite value
            "parameters": {},
            "processing_time": 0.0
        }
        
        filters = [
            ("gaussian_blur", lambda img: apply_per_channel(img, lambda c: gaussian_blur(c, size=kernel_size, sigma=blur_sigma))),
            ("laplacian_sharpen", lambda img: apply_per_channel(img, laplacian_sharpen)),
            ("adaptive_filter", lambda img: adaptive_blur_sharpen_color(img, blur_sigma=blur_sigma, ksize=kernel_size, var_threshold=var_threshold)),
            ("improved_adaptive", lambda img: improved_adaptive_filter(img, blur_sigma=blur_sigma, ksize=kernel_size, var_threshold_low=var_threshold_low, var_threshold_high=var_threshold_high))
        ]

        total_processing_time = 0
        for filter_type, func in filters:
            start_time = time.time()
            processed = func(original_image)
            filter_time = time.time() - start_time
            total_processing_time += filter_time
            psnr_value = psnr_color(original_image, processed)
            results[filter_type] = {
                "name": filter_type.replace('_', ' ').title(),
                "description": f"{filter_type.replace('_', ' ').title()} applied",
                "image": image_to_base64(processed),
                "psnr": psnr_value,
                "processing_time": round(filter_time, 2),
                "parameters": {
                    "blur_sigma": blur_sigma,
                    "kernel_size": kernel_size,
                    **({"var_threshold": var_threshold} if filter_type == "adaptive_filter" else {}),
                    **({"var_threshold_low": var_threshold_low, "var_threshold_high": var_threshold_high} if filter_type == "improved_adaptive" else {})
                }
            }

        summary = {
            "image_info": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "filename": file.filename,
                "file_size_mb": round(len(contents) / (1024 * 1024), 2)
            },
            "processing_results": {
                "total_filters": len(filters),
                "best_psnr": {
                    "filter": max([(k, v["psnr"]) for k, v in results.items() if k != "original"], key=lambda x: x[1])[0],
                    "value": round(max([(v["psnr"]) for k, v in results.items() if k != "original"]), 2)
                },
                "total_processing_time": round(total_processing_time, 2)
            }
        }
        
        logger.info(f"Successfully processed image: {file.filename}")
        
        return {
            "success": True,
            "summary": summary,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-single-filter")
async def process_single_filter(
    file: UploadFile = File(...),
    filter_type: str = Form("gaussian_blur"),
    blur_sigma: float = Form(1.0),
    kernel_size: int = Form(5),
    var_threshold: float = Form(500.0),
    var_threshold_low: float = Form(50.0),
    var_threshold_high: float = Form(1000.0)
):
    try:
        if kernel_size % 2 == 0:
            raise HTTPException(status_code=400, detail="Kernel size must be an odd number")
        if var_threshold_high <= var_threshold_low:
            raise HTTPException(status_code=400, detail="High threshold must be greater than low threshold")
        
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 10MB limit")
        
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Could not decode image. Please use JPG or PNG format")
        
        start_time = time.time()
        if filter_type == "gaussian_blur":
            processed = apply_per_channel(original_image, lambda img: gaussian_blur(img, size=kernel_size, sigma=blur_sigma))
        elif filter_type == "laplacian_sharpen":
            processed = apply_per_channel(original_image, laplacian_sharpen)
        elif filter_type == "adaptive_filter":
            processed = adaptive_blur_sharpen_color(original_image, blur_sigma=blur_sigma, ksize=kernel_size, var_threshold=var_threshold)
        elif filter_type == "improved_adaptive":
            processed = improved_adaptive_filter(original_image, blur_sigma=blur_sigma, ksize=kernel_size, var_threshold_low=var_threshold_low, var_threshold_high=var_threshold_high)
        else:
            raise HTTPException(status_code=400, detail="Invalid filter type. Available: gaussian_blur, laplacian_sharpen, adaptive_filter, improved_adaptive")
        
        processing_time = time.time() - start_time
        psnr_value = psnr_color(original_image, processed)
        
        return {
            "success": True,
            "filter_type": filter_type,
            "original_image": image_to_base64(original_image),
            "processed_image": image_to_base64(processed),
            "psnr": psnr_value,
            "processing_time": round(processing_time, 2),
            "image_info": {
                "width": original_image.shape[1],
                "height": original_image.shape[0],
                "channels": original_image.shape[2],
                "filename": file.filename,
                "file_size_mb": round(len(contents) / (1024 * 1024), 2)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing single filter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/available-filters")
async def get_available_filters():
    filters = {
        "gaussian_blur": {
            "name": "Gaussian Blur",
            "description": "Smoothing filter that reduces noise and detail",
            "parameters": {
                "blur_sigma": {"min": 0.1, "max": 5.0, "default": 1.0, "description": "Standard deviation for Gaussian kernel"},
                "kernel_size": {"min": 3, "max": 15, "default": 5, "description": "Size of the kernel (must be odd)"}
            }
        },
        "laplacian_sharpen": {
            "name": "Laplacian Sharpening", 
            "description": "Edge enhancement filter that increases detail and contrast",
            "parameters": {}
        },
        "adaptive_filter": {
            "name": "Adaptive Filter",
            "description": "Applies blur to smooth areas and sharpening to detailed areas",
            "parameters": {
                "blur_sigma": {"min": 0.1, "max": 5.0, "default": 1.0, "description": "Standard deviation for Gaussian kernel"},
                "kernel_size": {"min": 3, "max": 15, "default": 5, "description": "Size of the kernel (must be odd)"},
                "var_threshold": {"min": 10.0, "max": 2000.0, "default": 500.0, "description": "Variance threshold for switching between blur and sharpen"}
            }
        },
        "improved_adaptive": {
            "name": "Improved Adaptive Filter",
            "description": "Advanced adaptive filter with smooth blending between blur and sharpen",
            "parameters": {
                "blur_sigma": {"min": 0.1, "max": 5.0, "default": 1.0, "description": "Standard deviation for Gaussian kernel"},
                "kernel_size": {"min": 3, "max": 15, "default": 5, "description": "Size of the kernel (must be odd)"},
                "var_threshold_low": {"min": 1.0, "max": 500.0, "default": 50.0, "description": "Lower variance threshold"},
                "var_threshold_high": {"min": 100.0, "max": 5000.0, "default": 1000.0, "description": "Upper variance threshold"}
            }
        }
    }
    
    return {
        "success": True,
        "filters": filters,
        "general_limits": {
            "max_file_size_mb": 10,
            "supported_formats": ["JPG", "JPEG", "PNG"],
            "max_image_dimension": 4096
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
