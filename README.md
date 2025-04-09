# CUDA + GDAL Raster Processing (GLI)

This tool reads a 3‑band (RGB) raster using GDAL, computes the **Green Leaf Index (GLI)** for each pixel on the GPU via CUDA, and writes the result to a new GeoTIFF file.

---

## 📦 Prerequisites

- **CUDA Toolkit** (`nvcc`)  
- **GDAL** (headers and library)  
- A CUDA‑capable GPU  

---

## 🔧 Build

```
nvcc -o process_raster main.cu \
    -I/usr/include/gdal/ \
    -L/usr/lib/ -lgdal
```

---

## ▶️ Usage

```
./process_raster <input_file.tif> <output_file.tif>
```
