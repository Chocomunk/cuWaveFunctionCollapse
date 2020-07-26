#pragma once
#ifndef __CUDACC__
typedef unsigned int uint;

void __syncthreads();
int min(int x, int y);
int __popc(unsigned int x);

#endif
