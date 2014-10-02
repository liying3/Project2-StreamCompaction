#ifndef STREAMCOMPACTION_H
#define STREAMCOMPACTION_H
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\count.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <stdio.h>
#include <cmath>

#define BlockSize 128

void initial();

void prefixSumNaiveWarpper(int *A, int *R, int n);

void prefixSumSharedMemWarpper(int *A, int *R, int n);

void streamCompactionWarpper(int *A, int *R, int n);

void thrustStreamCompaction(int *A, int* &R, int n);

#endif