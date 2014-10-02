#include "StreamCompaction.h"
#include <iostream>

using namespace std;

__global__ void prefixSumNaive(int *in, int *out, int n, int base)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tx < n)
	{
		if (tx >= base)
			out[tx] = in[tx-base] + in[tx];
		else
			out[tx] = in[tx];
	}
}

void prefixSumNaiveWarpper(int *A, int *R, int N)
{
	int *in, *out;
	cudaMalloc((void**)&in, N * sizeof(int));
	cudaMemset(in, 0, sizeof(int));
	cudaMemcpy(in+1 , A, (N-1)* sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, N * sizeof(int));
	cudaMemset(out, 0, sizeof(int));
	cudaMemcpy(out+1 , A, (N-1)* sizeof(int), cudaMemcpyHostToDevice);

	float time;
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventRecord(startTime, 0);

	int base = 1;
	int	loops = log((double)BlockSize) / log(2.0);

	for (int i = 1; i <= loops; i++)
	{
		int *tem = in;
		in = out;
		out = in;

		prefixSumNaive<<<1, BlockSize>>>(in, out, N, base);
		cudaThreadSynchronize();
		base *= 2;
	}
	cudaEventCreate(&endTime);
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);

	cudaEventElapsedTime(&time, startTime, endTime);
	cudaEventDestroy(startTime);
	cudaEventDestroy(endTime);

	cout << "GPU(Naive): " << time << " ms" << endl;

	cudaMemcpy(R, out, N*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(in);
	cudaFree(out);
}

__global__ void prefixSumSharedMem(int *in, int *out, int n, int *blockSum, int base)
{
	int x = threadIdx.x;
	int tx = x + (blockIdx.x * blockDim.x);

	if (tx < n)
	{
		 __shared__ int outS[BlockSize];
		 __shared__ int inS[BlockSize];
		
		inS[x] = in[tx];
		__syncthreads();
		
		if (threadIdx.x >= base)
			outS[x] = inS[x-base] + inS[x];
		else
			outS[x] = inS[x];

		__syncthreads();

		out[tx] = outS[x];

		if (x == BlockSize-1)
			blockSum[blockIdx.x+1] = outS[x];
	}
}

__global__ void addBlockSum2All(int *in, int *out, int n, int *blockSum)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);

	 if (tx < n)
	 {
		 in[tx] += blockSum[blockIdx.x];
	 }
}

void prefixSumSharedMemWarpper(int *A, int *R, int N)
{
	int *in, *out;
	cudaMalloc((void**)&in, N * sizeof(int));
	cudaMemset(in, 0, sizeof(int));
	cudaMemcpy(in+1 , A, (N-1)* sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&out, N * sizeof(int));
	cudaMemset(out, 0, sizeof(int));
	cudaMemcpy(out+1 , A, (N-1)* sizeof(int), cudaMemcpyHostToDevice);

	int gridSize = ((int)(ceil(N / (float)BlockSize)));
	int *blockSum;
	cudaMalloc((void**)&blockSum, (gridSize+1)*sizeof(int));
	cudaMemset(blockSum, 0, sizeof(int));
	
	int *blockSumOut;
	cudaMalloc((void**)&blockSumOut, (gridSize+1)*sizeof(int));
	cudaMemset(blockSumOut, 0, sizeof(int));
	
	float time;
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventRecord(startTime, 0);

	int base = 1;
	int	loops= log((double)BlockSize) / log(2.0);

	for (int i = 1; i <= loops; i++)
	{
		int *tem = in;
		in = out;
		out = in;

		prefixSumSharedMem<<<gridSize, BlockSize>>>(in, out, N, blockSumOut, base);
		cudaThreadSynchronize();
		base *= 2;
	}
	
	if (gridSize > 1)
	{
		base = 1;
		loops= log((double)gridSize) / log(2.0);

		for (int i = 1; i <= loops; i++)
		{
			int *tem = blockSum;
			blockSum = blockSumOut;
			blockSumOut = tem;

			prefixSumNaive<<<1, BlockSize>>>(blockSum, blockSumOut, gridSize, base);
			cudaThreadSynchronize();
			base *= 2;
		}

		int *tem = in;
		in = out;
		out = tem;
		
		addBlockSum2All<<<gridSize, BlockSize>>>(in, out, N, blockSum);
		cudaThreadSynchronize();
	}
	cudaEventCreate(&endTime);
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);

	cudaEventElapsedTime(&time, startTime, endTime);
	cudaEventDestroy(startTime);
	cudaEventDestroy(endTime);
	cout << "GPU(Shared Memory): " << time << " ms" << endl;

	cudaMemcpy(R, out, N*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(in);
	cudaFree(out);
	cudaFree(blockSum);
	cudaFree(blockSumOut);
}

__global__ void scatter(int *A, int *R, int n)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tx < n)
	{
		if (A[tx] > 0)
			R[tx] = 1;
		else
			R[tx] = 0;
	}
}

__global__ void streamCompact(int *A, int *sum, int *R, int n)
{
	int tx = threadIdx.x + (blockIdx.x * blockDim.x);
	
	if (tx < n)
	{
		if (A[tx] > 0)
			R[sum[tx]] = A[tx];
	}
}

void streamCompactionWarpper(int *A, int *R, int N)
{
	int *Ad, *scatterVd;
	cudaMalloc((void**)&Ad, N * sizeof(int));
	cudaMemcpy(Ad, A, N* sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&scatterVd, N * sizeof(int));

	float time;
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventRecord(startTime, 0);

	int gridSize = ((int)(ceil(N / (float)BlockSize)));
	scatter<<<gridSize, BlockSize>>>(Ad, scatterVd, N);
	cudaThreadSynchronize();

	int *scatterV = new int[N];
	int *scatterSum = new int[N+1];
	cudaMemcpy(scatterV, scatterVd, N*sizeof(int), cudaMemcpyDeviceToHost);
	prefixSumSharedMemWarpper(scatterV, scatterSum, N+1);

	int returnSize = scatterSum[N];

	int *Rd, *Sumd;
	cudaMalloc((void**)&Rd, returnSize * sizeof(int));
	cudaMalloc((void**)&Sumd, (N+1) * sizeof(int));
	cudaMemcpy(Sumd, scatterSum, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	streamCompact<<<gridSize, BlockSize>>>(Ad, Sumd, Rd, N);

	cudaEventCreate(&endTime);
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);

	cudaEventElapsedTime(&time, startTime, endTime);
	cudaEventDestroy(startTime);
	cudaEventDestroy(endTime);
	cout << "GPU(Scatter): " << time << " ms" << endl;

	cudaMemcpy(R, Rd, returnSize*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(Ad);
	cudaFree(scatterVd);
	//cudaFree(scatterSum);
	//cudaFree(Rd);
}

struct is_not_zero
{
	__host__ __device__
	bool operator()(const int x)
	{
		return (x != 0);
	}
};

void thrustStreamCompaction(int* A, int* &R, int n)
{
	float time;
	cudaEvent_t startTime, endTime;
	cudaEventCreate(&startTime);
	cudaEventRecord(startTime, 0);

	int size = thrust::count_if(A, A+n, is_not_zero());
	R = new int[size];
	thrust::copy_if(A, A+n, R, is_not_zero());

	cudaEventCreate(&endTime);
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);

	cudaEventElapsedTime(&time, startTime, endTime);
	cudaEventDestroy(startTime);
	cudaEventDestroy(endTime);
	cout << "GPU(thrust): " << time << " ms" << endl;
}

