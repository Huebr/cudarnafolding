/*
============================================================================
/*
============================================================================
Name        : RNAFolding.cu

Author      : Pedro Jorge
Version     :
Copyright   : Your copyright notice
Description : CUDA compute reciprocals
============================================================================
*/

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//macros
#define MAX_FILENAME_SIZE 256
#define MAX_TEST_SIZE  5000
#define NBLOCK 4

//solve the RNA prediction problem
cudaError_t solverRNA(const char *, int *, int);

inline void printInfo(int *memo, const char* data, int size, int id) {
	FILE *fp;
	int i, j;
	char filename[MAX_FILENAME_SIZE];
	sprintf(filename, "output_info-%d.txt", id);
	fp = fopen(filename, "a");
	if (fp == NULL)
	{
		printf("Erro opening info file.");
		exit(1);
	}
	fprintf(fp, "--------------------new test---------------------\n");
	fprintf(fp, "Instance : %s\n", data);
	fprintf(fp, "Optimum value : %d\n", (memo[size - 1]));
	fprintf(fp, "Memoization Table : \n\n");
	for (i = 0; i < size; ++i) {
		for (j = 0; j < size; ++j) {
			fprintf(fp, "%d ", (memo[i*size + j]));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

__device__ void __gpu_sync(int goalVal, volatile int *Arrayin, volatile int *Arrayout){
	int tid_in_blk = threadIdx.x*blockDim.y + threadIdx.y;
	int nBlockNum = gridDim.x*gridDim.y;
	int  bid = blockIdx.x*gridDim.y + blockIdx.y;

	if (tid_in_blk == 0){
		Arrayin[bid] = goalVal;
	}

	if (bid == 1){
		if (tid_in_blk < nBlockNum){
			while (Arrayin[tid_in_blk] != goalVal){
			}
		}
		__syncthreads();

		if (tid_in_blk < nBlockNum){
			Arrayout[tid_in_blk] = goalVal;
		}

	}

	if (tid_in_blk == 0){
		while (Arrayout[bid] != goalVal){

		}
	}
	__syncthreads();

}

__device__ bool canPair(int base1, int base2) {
	bool case1, case2;
	case1 = (base1 == 67 && base2 == 71) || (base1 == 71 && base2 == 67);
	case2 = (base1 == 65 && base2 == 85) || (base1 == 85 && base2 == 65);
	return (case1 || case2);
}
__global__ void solverKernel(int *dev_data, int*dev_memo, int *dev_arrayIn, int *dev_arrayOut, int goalValue, int size)
{
	int i, j, opt;
	i = blockDim.x*blockIdx.x + threadIdx.x;
	for (int k = 5; k < size; k++){
		if (i<size - k){
			j = i + k;
			dev_memo[size*i + j] = dev_memo[size*i + (j - 1)];
			for (int t = i; t < j - 4; t++) {     //opt(i,j)=max(opt(i,j-1),1+opt(i,t-1)+opt(t+1,j-1))
				if (canPair(dev_data[t], dev_data[j])) {
					if (t == 0) {
						opt = 1 + dev_memo[size*(t + 1) + (j - 1)];
					}
					else {
						opt = 1 + dev_memo[i*size + (t - 1)] + dev_memo[size*(t + 1) + (j - 1)];
					}
					if (opt > dev_memo[size*i + j]) {
						dev_memo[i*size + j] = opt;
					}
				}
			}
		}
		__gpu_sync(goalValue, dev_arrayIn, dev_arrayOut);
		goalValue++;
	}
}
bool canPairCPU(char base1, char base2) {
	bool case1, case2;
	case1 = (base1 == 'C' && base2 == 'G') || (base1 == 'G' && base2 == 'C');
	case2 = (base1 == 'A' && base2 == 'U') || (base1 == 'U' && base2 == 'A');
	return (case1 || case2);
}
void findSolution(FILE *fp, const char* data, int *memo, int size, int i, int j){
	if (i<j - 4){
		if (memo[i*size + j] == memo[i*size + j - 1]){
			findSolution(fp, data, memo, size, i, j - 1);
		}
		else{
			for (int t = i; t<j - 4; t++){
				if (canPairCPU(data[t], data[j])){
					if (t == 0){
						if ((memo[i*size + j] - 1) == memo[(t + 1)*size + j - 1]){
							fprintf(fp, "%d %d undirected red\n", t, j);
							findSolution(fp, data, memo, size, t + 1, j - 1);
							break;
						}
					}
					else{
						if ((memo[i*size + j] - 1) == memo[i*size + t - 1] + memo[(t + 1)*size + j - 1]){
							fprintf(fp, "%d %d undirected red\n", t, j);
							findSolution(fp, data, memo, size, t + 1, j - 1);
							findSolution(fp, data, memo, size, i, t - 1);
							break;
						}
					}
				}
			}
		}
	}
}
inline void createVertices(int id, const char* data, int size){
	FILE *fileptr;
	char filename[MAX_FILENAME_SIZE];
	sprintf(filename, "vertices-%d.csv", id);
	fileptr = fopen(filename, "a");
	if (fileptr == NULL){
		printf("error opening vertices file.");
	}
	else{
		fprintf(fileptr, "Id Label\n");
		for (int i = 0; i<size; i++){
			fprintf(fileptr, "%d %c\n", i, data[i]);
		}
	}
	fclose(fileptr);
}


int main()
{
	FILE *input;
	char *filename;
	char testRNA[MAX_TEST_SIZE];
	int result;
	int id;
	id = 0;
	cudaError_t cudaStatus;

	//Memory Allocation to file name
	filename = (char*)malloc(MAX_FILENAME_SIZE*sizeof(char));

	//Reading filename
	printf("Write name of input file : ");
	scanf("%s", filename);

	//Open File to read input test data
	input = fopen(filename, "r");

	//Testing input opening
	if (input == NULL) {
		printf("Error opening file, please try again.");
		return 1;
	}

	printf("\n\n---------------- Begin Tests --------------------\n\n");

	//Begin reading file and testing
	while (fscanf(input, "%s", testRNA) != EOF) {
		id++;
		//launch solverRNA
		cudaStatus = solverRNA(testRNA, &result, id);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "solverRNA failed!");
			return 1;
		}

		printf("%s : ", testRNA);
		printf("%d base pairs.\n", result);
	}

	printf("\n\n---------------- Ending Tests --------------------\n\n");



	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	free(filename);
	system("pause");
	return 0;
}

// Helper function for using CUDA to solve RNA prediction in parallel with objective function maximum number of bases
cudaError_t solverRNA(const char *data, int *result, int id)
{

	int *dev_data = 0;//data in device
	int *dev_memo = 0;//memotable in device
	int *host_memo = 0;//memotable in host
	int *host_data = 0;
	int host_arrayIn[] = { 0, 0, 0, 0};
	int host_arrayOut[] = { 0, 0, 0, 0};
	int *dev_arrayIn;
	int *dev_arrayOut;
	int goalValue = 1;
	int size = strlen(data);
	FILE *solution;
	char solutionName[MAX_FILENAME_SIZE];
	const int size_memo = size*size;
	cudaError_t cudaStatus;

	//create events to record time elapsed
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//convert string to array of integers
	host_data = (int*)malloc(size*sizeof(int));
	for (int i = 0; i < size; ++i) host_data[i] = (int)data[i];
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	// Allocate	CPU buffer to memoTable
	host_memo = (int *)calloc(size_memo, sizeof(int));

	// Allocate GPU buffer to memoTable
	cudaStatus = cudaMalloc((void**)&dev_data, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_memo, size_memo*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_arrayIn, NBLOCK * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_arrayOut, NBLOCK * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_memo, host_memo, size_memo * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_data, host_data, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_arrayIn, host_arrayIn, NBLOCK * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_arrayOut, host_arrayOut, NBLOCK * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	cudaEventRecord(start);
	solverKernel << < 4, 1024 >> > (dev_data, dev_memo, dev_arrayIn, dev_arrayOut, goalValue, size);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "solverKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("solve in %f ms.\n", milliseconds);


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	//cudaStatus = cudaDeviceSynchronize();
	//if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching solverKernel!\n", cudaStatus);
		//goto Error;
	//}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(host_memo, dev_memo, size_memo*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	*result = host_memo[size - 1];
	printInfo(host_memo, data, size, id);
	createVertices(id, data, size);
	sprintf(solutionName, "edges-%d.csv", id);
	solution = fopen(solutionName, "a");
	if (solution == NULL){
		printf("error writing output connections.\n");
	}
	fprintf(solution, "Source Target Type Color\n");
	for (int i = 0; i<size - 1; i++){
		fprintf(solution, "%d %d undirected black\n", i, i + 1);
	}
	findSolution(solution, data, host_memo, size, 0, size - 1);
	fclose(solution);
Error:
	cudaFree(dev_memo);
	cudaFree(dev_data);
	cudaFree(dev_arrayIn);
	cudaFree(dev_arrayOut);
	free(host_data);
	free(host_memo);
	return cudaStatus;
}
