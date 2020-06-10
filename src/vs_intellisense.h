#pragma once
#ifndef __CUDACC__
typedef unsigned int uint;

void __syncthreads();
void atomicAdd(int* addr, int val);

#endif
