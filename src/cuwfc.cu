#include "cuda_runtime.h"

// Imports for developing on windows
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "vs_intellisense.h"
#endif

#include "cuwfc.cuh"

#define NUM_THREADS_1D(x) ((x) > 1024 ? 1024 : (x))
#define NUM_BLOCKS_1D(total_threads, num_threads) ((total_threads) / (num_threads))

/*
 * Device Kernels
 */

__global__
void cudaCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
	unsigned tid = threadIdx.x;
	while (tid < num_patterns) {
		// Set 1 if it is the intended state, 0 otherwise.
		waves[idx + tid] = tid == state;
		
		tid += blockDim.x * gridDim.x;
	}
}

/*
 * Host Interface Functions
 */

// NOTE: Does not update entropy value, only updates waves
void cudaCallCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
	int numThreads = NUM_THREADS_1D(num_patterns);
	int numBlocks = NUM_BLOCKS_1D(num_patterns, numThreads);
	cudaCollapseWaveKernel<<<numBlocks, numThreads>>>(waves, idx, state, num_patterns);
}

