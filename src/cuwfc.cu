#include "cuda_runtime.h"
#include "helper_cuda.h"

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
 void cudaClearKernel(char* waves, int* entropies, int num_waves, int num_patterns) {

	 unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
	 int tid_base = tid * num_patterns;

	 while (tid < num_waves) {
		for (int patt = 0; patt < num_patterns; patt++) {

			waves[tid_base + patt] = true;
		}
	
		entropies[tid] = num_patterns;

		tid += blockDim.x * gridDim.x;
		tid_base = tid * num_patterns;
	}
}

__global__ 
void cudaLowestEntropyKernel(int* entropies, int num_waves, int* lowest_entropy_idx) { 
	// index of lowest entropy is "returned" to the lowest_entropy_idx variable.

    extern __shared__ float shmem[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;


	// Use each thread to transfer global entropy data into shared memory.
	for (; idx < num_waves; idx += blockDim.x * gridDim.x) {
		shmem[tid] = entropies[tid];
	}
	__syncthreads(); 
	
	/*
	 * Do a reduction 
	 * our operation : storing the minimum value (> 1) betweeen the two splits
	 * NOTE: Consider any wave with entropy <= 1 to be collapsed
	 */
	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1) {

		if (tid < j) {
			/* 
			 * Update lower half if:
			 * the lower entropy is in second half OR 
			 * entropy in first half is collapsed,
			 *
			 * AND the second half entropy is not collapsed
			 */
			if ((shmem[tid] > shmem[tid + j] || shmem[tid] <= 1) && shmem[tid + j] > 1) {
				shmem[tid] = shmem[tid + j];
				// store index as well?
			}
			// if both are collapsed, then mark the corresponding index as -1
			else if (shmem[tid] <= 1 && shmem[tid + j] <= 1) {
				shmem[tid] = -1;
			}
			// otherwise if the lower of the two entropies is in the first half, 
			// keep it 
		}
		__syncthreads();
	}
	// shmem[0] should store the lowest entropy if there exists an uncollapsed wave,
	// if all waves are collapsed, shmem[0] should be set to -1

	
	if (tid == 0){
		// Storing index of into global variable passed into the kernel via 
		// atomicAdd
		atomicAdd(lowest_entropy_idx, shmem[0]);
	}

}

template <unsigned int blockSize>
__device__ void warpReduceOr(volatile float* sdata, unsigned int tid) {
	volatile float* addr = sdata + tid;
	if (blockSize >= 64) sdata[tid] = *addr || *(addr + 32);
	if (blockSize >= 32) sdata[tid] = *addr || *(addr + 16);
	if (blockSize >= 16) sdata[tid] = *addr || *(addr + 8);
	if (blockSize >= 8) sdata[tid] = *addr || *(addr + 4);
	if (blockSize >= 4) sdata[tid] = *addr || *(addr + 2);
	if (blockSize >= 2) sdata[tid] = *addr || *(addr + 1);
}

template <unsigned int blockSize>
__global__
void cudaReduceOrKernel(bool* out_data, int* output, int length) {
	__shared__ float data[blockSize];
	unsigned tid = threadIdx.x;
	unsigned get_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned gridSize = blockSize * 2 * gridDim.x;

	float* addr = data + tid;   // Address of the shmem element for this thread
	data[tid] = out_data[get_idx];
	data[tid] = *addr || out_data[get_idx + blockSize];
	get_idx += gridSize;

	while (get_idx < length) {
		data[tid] = *addr || out_data[get_idx];
		data[tid] = *addr || out_data[get_idx + blockSize];
		get_idx += gridSize;
	}
	__syncthreads();

	if (blockSize >= 1024) { if (tid < 512) { data[tid] = *addr || *(addr + 512); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { data[tid] = *addr || *(addr + 256); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { data[tid] = *addr || *(addr + 128); } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { data[tid] = *addr || *(addr + 64); } __syncthreads(); }

	if (tid < 32) warpReduceOr<blockSize>(data, tid);
	if (tid == 0) atomicOr(output, data[0]);
}

__device__
void copyFits(bool* s_data, bool* fits, int length) {
	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < length) {
		s_data[tid] = fits[tid];
		
		tid += blockDim.x * gridDim.x;
	}
}

__global__
void cudaUpdateWavesKernel(char* waves, bool* fits, int* overlays, bool* changes,
								int waves_x, int waves_y, 
								int num_patterns, int num_overlays) {
	// TODO: Use shared memory
	extern __shared__ bool s_fits[];

	copyFits(s_fits, fits, num_patterns * num_overlays * num_patterns);
	
	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned tid_base = tid * num_patterns;
	bool changed = false;
	
	while(tid < waves_x * waves_y) {
		for (int c=0; c < num_patterns; c++) {

			int c_idx = c * num_overlays * num_patterns;
			bool allowed = false;
			for (int o=0; o < num_overlays; o++) {
				int o_x = overlays[2 * o];
				int o_y = overlays[2 * o + 1];
				int o_idx = o * num_patterns;
				int other_idx = tid + o_y * waves_x + o_x;
				int other_base = other_idx * num_patterns;
				bool valid = other_idx >= 0 && other_idx < waves_x * waves_y;
				
				for (int other_patt=0; other_patt < num_patterns; other_patt++) {
					// Splits conditions to encourage cache hits.
					// Note: center wave condition can be computed externally
					// and block all both inner loops, but in the interest of
					// preventing thread divergence, the inner loops are kept as
					// redundant computations.
					bool waves_cond = valid && waves[tid_base + c] && waves[other_base + other_patt];
					bool is_fit = s_fits[c_idx + o_idx + other_patt];

					// We can also just stop once "allowed = true", but again we
					// want to prevent thread divergence.
					
					// A fit has been found, so this state is allowed for now.
					__syncthreads();
					allowed = allowed || (waves_cond && is_fit);
				}
			}

			// A state that was once allowed has been dis-allowed
			changed = changed || (waves[tid_base + c] && !allowed);
			__syncthreads();
			waves[tid_base + c] = allowed;
		}

		__syncthreads();
		changes[tid] = changed;

		tid += blockDim.x * gridDim.x;
		tid_base = tid * num_patterns;
	}
}

__global__
void cudaCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < num_patterns) {
		// Set 1 if it is the intended state, 0 otherwise.
		waves[idx + tid] = tid == state;
		
		tid += blockDim.x * gridDim.x;
	}
}

__global__
void cudaComputeEntropiesKernel(char* waves, int* entropies, int num_waves, int num_patterns) {
	unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned tid_base = tid * num_patterns;
	while (tid < num_waves) {
		int total = 0;
		for (int i=0; i < num_patterns; i++) {
			total += waves[tid_base + i];
		}

		__syncthreads();
		entropies[tid] = total;
		
		tid += blockDim.x * gridDim.x;
	}
}

/*
 * Host Interface Functions
 */

 void cudaCallClearKernel(char* waves, int* entropies, int num_waves, int num_patterns) {
	int numThreads = NUM_THREADS_1D(num_waves);
	int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
	cudaClearKernel<<<numBlocks, numThreads>>>(waves, entropies, num_waves, num_patterns);
 }

void cudaCallUpdateWavesKernel(char* waves, bool* fits, int* overlays,
								int waves_x, int waves_y, 
								int num_patterns, int num_overlays,
								bool* changes, int changes_size, int* changed) {
	int total_work = waves_x * waves_y;
	int numThreads = NUM_THREADS_1D(total_work);
	int numBlocks = NUM_BLOCKS_1D(total_work, numThreads);
	int fits_size = num_patterns * num_overlays * num_patterns;
	
	cudaUpdateWavesKernel<<<numBlocks, numThreads, fits_size>>>(
		waves, fits, overlays, changes, 
		waves_x, waves_y, num_patterns, num_overlays);

	switch (numThreads) {
	case 1024: cudaReduceOrKernel<1024> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case 512: cudaReduceOrKernel<512> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case 256: cudaReduceOrKernel<256> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case 128: cudaReduceOrKernel<128> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case  64: cudaReduceOrKernel< 64> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case  32: cudaReduceOrKernel< 32> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case  16: cudaReduceOrKernel< 16> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case   8: cudaReduceOrKernel<  8> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case   4: cudaReduceOrKernel<  4> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case   2: cudaReduceOrKernel<  2> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	case   1: cudaReduceOrKernel<  1> <<<numBlocks, numThreads>>> (changes, changed, changes_size); break;
	}
}

 void cudaCallLowestEntropyKernel(int* entropies, int num_waves, int num_patterns, int* lowest_entropy_idx) { 
	int numThreads = NUM_THREADS_1D(num_waves);
	int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
	cudaLowestEntropyKernel<<<numBlocks, numThreads>>>(entropies, num_waves, lowest_entropy_idx);
 }

// NOTE: Does not update entropy value, only updates waves
void cudaCallCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
	int numThreads = NUM_THREADS_1D(num_patterns);
	int numBlocks = NUM_BLOCKS_1D(num_patterns, numThreads);
	cudaCollapseWaveKernel<<<numBlocks, numThreads>>>(waves, idx, state, num_patterns);
}

void cudaCallComputeEntropiesKernel(char* waves, int* entropies, int num_waves, int num_patterns) {
	int numThreads = NUM_THREADS_1D(num_waves);
	int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
	cudaComputeEntropiesKernel<<<numBlocks, numThreads>>>(waves, entropies, num_waves, num_patterns);
}

