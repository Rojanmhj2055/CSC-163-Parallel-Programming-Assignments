#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>

#define TILE_WIDTH 16 	//do not change this value

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows,int numBColumns,int numCRows, int numCColumns) {
  //@@ Insert code to implement tiled matrix multiplication here
  //@@ You have to use shared memory to write this kernel

    __shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float v = 0.0;

    for (int i = 0; i < (int)(ceil((float)numAColumns / TILE_WIDTH)); i++)
    {
        if (i * TILE_WIDTH + tx < numAColumns && row < numARows)
            sharedM[ty][tx] = A[row * numAColumns + i * TILE_WIDTH + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i * TILE_WIDTH + ty < numBRows && col < numBColumns)
            sharedN[ty][tx] = B[(i * TILE_WIDTH + ty) * numBColumns + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for (int j = 0; j < TILE_WIDTH; j++)
            v += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns)
        C[row * numCColumns + col] = v;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = 0;
  numCColumns = 0;
  if (numAColumns != numBRows) {
      wbLog(TRACE, "numAColumns != numBRows, Break ");
      return 1;
  }
  numCRows = numARows;
  numCColumns = numBColumns;
  unsigned int A_size = numARows * numAColumns * sizeof(float);
  unsigned int B_size = numBRows * numBColumns * sizeof(float);
  unsigned int C_size = numCRows * numCColumns * sizeof(float);
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(C_size);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, A_size);
  cudaMalloc((void**)&deviceB, B_size);
  cudaMalloc((void**)&deviceC, C_size);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, C_size, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((int)ceil(float(numCColumns ) / TILE_WIDTH), (int)ceil(float(numCRows) / TILE_WIDTH), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, C_size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
