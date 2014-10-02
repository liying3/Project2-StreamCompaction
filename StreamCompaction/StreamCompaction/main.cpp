#include "StreamCompaction.h"
#include <time.h>

using namespace std;

#define N 200

void serialPrefixSum(int *A, int *R, int n);

void serialScatter(int *A, int *R, int n);

void serialStreamCompaction(int *A, int* &R, int n);

int main()
{
	//int A[10] = {0,0,3,4,0,6,6,7,0,1};
	//int A[7] = {1,2,3,4,5,6,7};

	int *A = new int[N];
	for (int i = 0; i < N; i++)
		A[i] = rand() % 10;

	cout << "----------------------------Prefix Sum----------------------------------" << endl;
	int *rCPU = new int[N+1];
	serialPrefixSum(A, rCPU, N+1);

	int *rGPUnaive = new int[N+1];
	prefixSumNaiveWarpper(A, rGPUnaive, N+1);

	int *rGPUshared = new int[N+1];
	prefixSumSharedMemWarpper(A, rGPUshared, N+1);

	cout << "--------------------------Stream Compaction------------------------------" << endl;
	
	int *rCPUsc = NULL;
	serialStreamCompaction(A, rCPUsc, N+1);
	
	int *rGPUsc = new int[N];
	streamCompactionWarpper(A, rGPUsc, N);
	
	int *rThrustsc = NULL;
	thrustStreamCompaction(A, rThrustsc, N);
	
	return 0;
}

void serialPrefixSum(int *A, int *R, int n)
{
	clock_t startTime, endTime;
	double tCPU;
	startTime = clock();

	R[0] = 0;
	for (int i = 1; i < n; i++)
		R[i] = A[i-1] + R[i-1];

	endTime = clock();
	tCPU = ((double)(endTime - startTime )) / CLOCKS_PER_SEC *1000; //ms
	cout << "CPU: " << tCPU << " ms" << endl; 
}

void serialScatter(int *A, int *R, int n)
{
	int *v = new int[n];

	for (int i = 0; i < n; i++)
	{
		if (A[i] > 0)
			v[i] = 1;
		else v[i] = 0;
	}
	serialPrefixSum(v, R, n);
}

void serialStreamCompaction(int *A, int* &R, int n)
{
	clock_t startTime, endTime;
	double tCPU;
	startTime = clock();

	int *scatterSum = new int[n];
	serialScatter(A, scatterSum, n);
	
	int compactSize = scatterSum[n-1];
	R = new int[compactSize];
	int j = 0;
	for (int i = 0; i < n; i++)
	{
		if (scatterSum[i] == j+1)
			R[j++] = A[i-1];
	}

	endTime = clock();
	tCPU = ((double)(endTime - startTime )) / CLOCKS_PER_SEC *1000; //ms
	cout << "CPU: " << tCPU << " ms" << endl; 
}