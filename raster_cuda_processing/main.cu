#include <iostream>
#include <gdal_priv.h>
#include <cuda_runtime.h>

__global__ void processRaster(float *data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = data[idx] * 2.0f; 
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

    GDALDataset *poDataset = (GDALDataset *) GDALOpen(inputFile, GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "Falha ao abrir o arquivo raster." << std::endl;
        return 1;
    }

    int width = poDataset->GetRasterXSize();
    int height = poDataset->GetRasterYSize();
    int bands = poDataset->GetRasterCount();

    if (bands < 1) {
        std::cerr << "O raster não tem bandas." << std::endl;
        GDALClose(poDataset);
        return 1;
    }

    GDALRasterBand *poBand = poDataset->GetRasterBand(1);
    float *hData = (float *) CPLMalloc(sizeof(float) * width * height);
    poBand->RasterIO(GF_Read, 0, 0, width, height, hData, width, height, GDT_Float32, 0, 0);

    float *dData;
    cudaMalloc((void **)&dData, sizeof(float) * width * height);
    cudaMemcpy(dData, hData, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    processRaster<<<numBlocks, threadsPerBlock>>>(dData, width, height);

    cudaMemcpy(hData, dData, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset *poDstDataset = poDriver->Create(outputFile, width, height, 1, GDT_Float32, nullptr);

    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    poDstDataset->SetGeoTransform(adfGeoTransform);
    const char *pszProjection = poDataset->GetProjectionRef();
    poDstDataset->SetProjection(pszProjection);

    GDALRasterBand *poDstBand = poDstDataset->GetRasterBand(1);
    poDstBand->RasterIO(GF_Write, 0, 0, width, height, hData, width, height, GDT_Float32, 0, 0);

    CPLFree(hData);
    cudaFree(dData);
    GDALClose(poDataset);
    GDALClose(poDstDataset);

    return 0;
}
