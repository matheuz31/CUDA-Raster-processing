#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>

__global__ void calculateGLI(float *dRed, float *dGreen, float *dBlue, float *dGLI, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        float green = dGreen[idx];
        float red = dRed[idx];
        float blue = dBlue[idx];
        dGLI[idx] = (2.0f * green - red - blue) / (2.0f * green + red + blue);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    GDALAllRegister(); 

    GDALDataset *poDataset = (GDALDataset *) GDALOpen(inputFile, GA_ReadOnly); //Cast to GDALDataset
    if (poDataset == nullptr) {
        std::cerr << "failed to open raster file" << std::endl;
        return 1;
    }

    int width = poDataset->GetRasterXSize();
    int height = poDataset->GetRasterYSize();
    int bands = poDataset->GetRasterCount();

    if (bands < 3) {
        std::cerr << "not enough arguments." << std::endl;
        GDALClose(poDataset);
        return 1;
    }

    // Process in blocks
    const int blockSize = 1024; // Adjust block size as needed
    float *hRed = (float *) CPLMalloc(sizeof(float) * blockSize * blockSize);
    float *hGreen = (float *) CPLMalloc(sizeof(float) * blockSize * blockSize);
    float *hBlue = (float *) CPLMalloc(sizeof(float) * blockSize * blockSize);
    float *hGLI = (float *) CPLMalloc(sizeof(float) * blockSize * blockSize);

    float *dRed, *dGreen, *dBlue, *dGLI;
    cudaMalloc((void **)&dRed, sizeof(float) * blockSize * blockSize);
    cudaMalloc((void **)&dGreen, sizeof(float) * blockSize * blockSize);
    cudaMalloc((void **)&dBlue, sizeof(float) * blockSize * blockSize);
    cudaMalloc((void **)&dGLI, sizeof(float) * blockSize * blockSize);

    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff"); //GetGDALDriverManager returns a pointer to a singleton then return the drive for the Gtiff type
    GDALDataset *poDstDataset = poDriver->Create(outputFile, width, height, 1, GDT_Float32, nullptr); //creates a new GDALDataset

    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    poDstDataset->SetGeoTransform(adfGeoTransform);
    const char *pszProjection = poDataset->GetProjectionRef();
    poDstDataset->SetProjection(pszProjection);

    for (int y = 0; y < height; y += blockSize) {
        for (int x = 0; x < width; x += blockSize) {
            int xSize = std::min(blockSize, width - x);
            int ySize = std::min(blockSize, height - y);

            GDALRasterBand *poRedBand = poDataset->GetRasterBand(1);
            GDALRasterBand *poGreenBand = poDataset->GetRasterBand(2);
            GDALRasterBand *poBlueBand = poDataset->GetRasterBand(3);

            poRedBand->RasterIO(GF_Read, x, y, xSize, ySize, hRed, xSize, ySize, GDT_Float32, 0, 0);
            poGreenBand->RasterIO(GF_Read, x, y, xSize, ySize, hGreen, xSize, ySize, GDT_Float32, 0, 0);
            poBlueBand->RasterIO(GF_Read, x, y, xSize, ySize, hBlue, xSize, ySize, GDT_Float32, 0, 0);

            cudaMemcpy(dRed, hRed, sizeof(float) * xSize * ySize, cudaMemcpyHostToDevice);
            cudaMemcpy(dGreen, hGreen, sizeof(float) * xSize * ySize, cudaMemcpyHostToDevice);
            cudaMemcpy(dBlue, hBlue, sizeof(float) * xSize * ySize, cudaMemcpyHostToDevice);

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((xSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (ySize + threadsPerBlock.y - 1) / threadsPerBlock.y);

            calculateGLI<<<numBlocks, threadsPerBlock>>>(dRed, dGreen, dBlue, dGLI, xSize, ySize);

            cudaMemcpy(hGLI, dGLI, sizeof(float) * xSize * ySize, cudaMemcpyDeviceToHost);

            GDALRasterBand *poDstBand = poDstDataset->GetRasterBand(1);
            poDstBand->RasterIO(GF_Write, x, y, xSize, ySize, hGLI, xSize, ySize, GDT_Float32, 0, 0);
        }
    }

    CPLFree(hRed);
    CPLFree(hGreen);
    CPLFree(hBlue);
    CPLFree(hGLI);
    cudaFree(dRed);
    cudaFree(dGreen);
    cudaFree(dBlue);
    cudaFree(dGLI);
    GDALClose(poDataset);
    GDALClose(poDstDataset);

    return 0;
}
