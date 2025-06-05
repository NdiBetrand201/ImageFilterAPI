# ImageFilterAPI
A FastAPI-powered image processing API with a Flutter frontend for applying advanced filters like Gaussian Blur, Laplacian Sharpening, and adaptive blending. Upload images, tweak parameters, and get enhanced results with PSNR metrics. Deployed on Render, ideal for real-time image enhancement.


# ImageFilterAPI

A powerful image processing application combining a **FastAPI** backend and a **Flutter** frontend. This project allows users to upload images, apply advanced filters (Gaussian Blur, Laplacian Sharpening, Adaptive Filter, and Improved Adaptive Filter), and receive enhanced images with PSNR quality metrics. Deployed on Render, it’s designed for real-time image enhancement with a modern, user-friendly interface.

## Features
- **Backend (FastAPI)**:
  - Processes images with OpenCV and NumPy for high-quality filtering.
  - Supports endpoints for multiple filters (`/process-image`), single filters (`/process-single-filter`), and filter metadata (`/available-filters`).
  - Returns base64-encoded images and PSNR metrics for quality assessment.
- **Frontend (Flutter)**:
  - Sleek UI with screens for image selection, parameter tuning, results display, and full-screen image viewing.
  - Supports camera/gallery image uploads and real-time parameter adjustments.
  - Responsive design with gradient themes and animations.
- **Deployment**: Hosted on Render for scalable, cloud-based access.
- **CORS**: Configured for secure integration with Flutter clients.

## Tech Stack
- **Backend**: FastAPI, OpenCV, NumPy, Pillow, Uvicorn
- **Frontend**: Flutter, Provider, http, image_picker
- **Deployment**: Render
- **Others**: Python, Dart, Git

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/NdiBetrand201/ImageFilterAPI.git
