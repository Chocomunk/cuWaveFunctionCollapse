#include "cuda_runtime.h"
#include "helper_cuda.h"

// Imports for developing on windows
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <device_launch_parameters.h>
#include "vs_intellisense.h"
#endif

#include "cuwfc.cuh"

// The minimum shared memory size should be 64 for an efficient reduction.
// Otherwise, we go out of bounds when reducing within the warp.
#define SHMEM_SIZE(blocksize) ((blocksize) < 64 ? 64 : blocksize)

#define I_MIN(x,y) ((x < y) ? x : y)

// MIN and MIN_IDX implementations that are limited to staying above 1 (for entropy)
#define L_MIN(x,y) ((x <= 1) ? y : ((y <= 1) ? x : ((x < y) ? x : y)))
#define L_MIN_IDX(x,y,x_i,y_i) ((x <= 1) ? y_i : ((y <= 1) ? x_i : ((x < y) ? x_i : y_i)))

// TODO: make this architecture friendly by checking architecture for max threads.
#define NUM_THREADS_1D(x) ((x) > 512 ? 512 : (x))
#define NUM_BLOCKS_1D(total_threads, num_threads) ((total_threads + num_threads - 1) / (num_threads))

namespace wfc
{
		
////////////////////////////////////////////////////////////////////////////////
///																			 ///
///								Device Kernels								 ///
///																			 ///
////////////////////////////////////////////////////////////////////////////////

	 __global__ 
	 void cudaClearKernel(int* waves, int* entropies, int num_waves, int num_patterns, int num_patt_ints) {

		 unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
		 int tid_base = tid * num_patt_ints;

		// Reset all tiles to perfect superposition and update entropy to reflect this.
		 while (tid < num_waves) {
			int patt_int_idx = 0;
			for (int pat_idx = 0; pat_idx < num_patterns; pat_idx += INT_BITS) {
				// If remaining patterns > INT_BITS, then set all bits in this int
				// Otherwise, only set the bits for the remaining patterns.
				int bitv_size = MIN((num_patterns - pat_idx), INT_BITS);
				waves[tid_base + patt_int_idx] = (uint32_t)(((uint64_t)1 << bitv_size) - 1);
				patt_int_idx++;
			}

			__syncthreads();
			entropies[tid] = num_patterns;

			tid += blockDim.x * gridDim.x;
			tid_base = tid * num_patt_ints;
		}
	}

	__device__
	void setMin(volatile int* smin, volatile int* sidx, unsigned int tid, unsigned int shift) {
		 volatile int* addr_m = smin + tid;   // Address of the shmem element for this thread
		 volatile int* addr_i = sidx + tid;   // Address of the shmem element for this thread

		// Set the index first or else the min points somewhere else
		 sidx[tid] = L_MIN_IDX(*addr_m, *(addr_m +shift), *(addr_i), *(addr_i + shift));
		 smin[tid] = L_MIN(*addr_m, *(addr_m + shift));
	 }

	 template <unsigned int blockSize>
	 __device__ void warpReduceMin(volatile int* smin, volatile int* sidx, unsigned int tid) {
		 if (blockSize >= 64) setMin(smin, sidx, tid, 32);
		 if (blockSize >= 32) setMin(smin, sidx, tid, 16);
		 if (blockSize >= 16) setMin(smin, sidx, tid, 8);
		 if (blockSize >= 8) setMin(smin, sidx, tid, 4);
		 if (blockSize >= 4) setMin(smin, sidx, tid, 2);
		 if (blockSize >= 2) setMin(smin, sidx, tid, 1);
	 }

	 template <unsigned int blockSize>
	 __global__
	 void cudaLowestEntropyKernel(int* entropies, int* workspace, int num_waves) {

		 __shared__ int s_entropy[SHMEM_SIZE(blockSize)];
		 __shared__ int s_indices[SHMEM_SIZE(blockSize)];
		 unsigned tid = threadIdx.x;
		 unsigned get_idx = blockIdx.x * blockDim.x + threadIdx.x;
		 unsigned gridSize = blockSize * 2 * gridDim.x;

		// Pre-collect the first two accumulations
		 int min = entropies[get_idx];
		 int min_idx = get_idx;
		if (get_idx + blockSize < num_waves) {
			 min_idx = L_MIN_IDX(min, entropies[get_idx + blockSize],
				 get_idx, get_idx + blockSize);
			 min = L_MIN(min, entropies[get_idx + blockSize]);
		}
		 get_idx += gridSize;

		// Accumulate the rest of the data into shared memory.
		 while (get_idx < num_waves) {
			 min_idx = L_MIN_IDX(min, entropies[get_idx], min_idx, get_idx);
			 min = L_MIN(min, entropies[get_idx]);
			
			 min_idx = L_MIN_IDX(min, entropies[get_idx + blockSize], min_idx, get_idx + blockSize);
			 min = L_MIN(min, entropies[get_idx + blockSize]);
			 get_idx += gridSize;
		 }
		 s_entropy[tid] = min;
		 s_indices[tid] = min_idx;
		 __syncthreads();

		// Now compute the reduction over the shared memory.
		 if (blockSize >= 1024) { if (tid < 512) { setMin(s_entropy, s_indices, tid, 512); } __syncthreads(); }
		 if (blockSize >= 512) { if (tid < 256) { setMin(s_entropy, s_indices, tid, 256); } __syncthreads(); }
		 if (blockSize >= 256) { if (tid < 128) { setMin(s_entropy, s_indices, tid, 128); } __syncthreads(); }
		 if (blockSize >= 128) { if (tid < 64) { setMin(s_entropy, s_indices, tid, 64); } __syncthreads(); }

		 if (tid < 32) warpReduceMin<blockSize>(s_entropy, s_indices, tid);

		// Update the workspace with the block-minima
		 if (tid == 0) {
			 workspace[2 * blockIdx.x] = s_entropy[0];
			 workspace[2 * blockIdx.x + 1] = s_indices[0];
		 }
	 }

	__global__
	void cudaReduceMinIdxKernel(int* workspace, int length) {
		/* The workspace is structured as (with padding on the end):
		 *	[min_0, idx_0, min_1, idx_1, ..., min_n, idx_n, 0, 0, ..., 0]
		 */

		// Split shared memory into a min array and idx array
		extern __shared__ int smin[];
		int* sidx = smin + length;
		
		unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned tid = threadIdx.x;

		// Pre-accumulate the minimums into shared memory
		int min = 0;
		int min_idx = -1;
		for (; idx < length; idx += blockDim.x * gridDim.x) {
			min_idx = L_MIN_IDX(min, workspace[idx], min_idx, idx);
			min = L_MIN(min, workspace[idx]);
		}
		smin[tid] = min;
		sidx[tid] = min_idx;
		__syncthreads();

		// Do the reduction to find the min over the shared memory
		for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s)
				setMin(smin, sidx, tid, s);
			__syncthreads();
		}


		// Update the workspace with the new block minima
		if (tid == 0) {
			 workspace[2 * blockIdx.x] = smin[0];
			 workspace[2 * blockIdx.x + 1] = sidx[0];
		}
	 }

	template <unsigned int blockSize>
	__device__ void warpReduceOr(volatile int* sdata, unsigned int tid) {
		volatile int* addr = sdata + tid;
		if (blockSize >= 64) sdata[tid] = *addr || *(addr + 32);
		if (blockSize >= 32) sdata[tid] = *addr || *(addr + 16);
		if (blockSize >= 16) sdata[tid] = *addr || *(addr + 8);
		if (blockSize >= 8) sdata[tid] = *addr || *(addr + 4);
		if (blockSize >= 4) sdata[tid] = *addr || *(addr + 2);
		if (blockSize >= 2) sdata[tid] = *addr || *(addr + 1);
	}

	template <unsigned int blockSize>
	__global__
	void cudaReduceOrKernel(int* in_data, int* output, int length) {
		__shared__ int data[SHMEM_SIZE(blockSize)];
		unsigned tid = threadIdx.x;
		unsigned get_idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned gridSize = blockSize * 2 * gridDim.x;

		// Pre-collect the first two accumulations
		int* addr = data + tid;   // Address of the shmem element for this thread
		data[tid] = in_data[get_idx];
		if (get_idx + blockSize < length)
			data[tid] = *addr || in_data[get_idx + blockSize];
		get_idx += gridSize;

		// Accumulate the rest of the data into shared memory.
		while (get_idx < length) {
			data[tid] = *addr || in_data[get_idx];
			data[tid] = *addr || in_data[get_idx + blockSize];
			get_idx += gridSize;
		}
		__syncthreads();

		// Now compute the reduction over the shared memory.
		if (blockSize >= 1024) { if (tid < 512) { data[tid] = *addr || *(addr + 512); } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { data[tid] = *addr || *(addr + 256); } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { data[tid] = *addr || *(addr + 128); } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { data[tid] = *addr || *(addr + 64); } __syncthreads(); }

		if (tid < 32) warpReduceOr<blockSize>(data, tid);

		// Atomically accumulate over all blocks. Can change this to a recursive
		// reduction by updating the in_data
		if (tid == 0) atomicOr(output, data[0]);
	}

	__global__
	void _cudaUpdateWavesKernel(int* waves, int* fits, int* overlays, int* workspace,
									int waves_x, int waves_y, 
									int num_patterns, int num_overlays, int num_pat_ints) {
		extern __shared__ int s_waves[];
		int* s_other = s_waves + blockDim.x * num_pat_ints * sizeof(int);
	 	int* s_masks = s_other + blockDim.x * num_pat_ints * sizeof(int);

		unsigned idx_base = threadIdx.x * num_pat_ints;
		unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned tid_base = tid * num_pat_ints;

	 	// TODO: Experiment with __syncthreads()
		while (tid < waves_x * waves_y) {

			// Update center tile state in shared memory
			for (int i=0; i < num_pat_ints; i++)
				s_waves[idx_base + i] = waves[tid_base + i];

			__syncthreads();

			// Loop over overlays between the center and neighbor
			for (int o=0; o < num_overlays; o++) {
				int o_x = overlays[2 * o];
				int o_y = overlays[2 * o + 1];
				int other_idx = tid + o_x * waves_x + o_y;
				int other_base = other_idx * num_pat_ints;
				bool valid = other_idx >= 0 && other_idx < waves_x * waves_y;

				__syncthreads();

				// Initialize mask and cache neighbor state
				for (int i=0; i < num_pat_ints; i++) {
					s_other[idx_base + i] = valid ? waves[other_base + i] : 0;
					s_masks[idx_base + i] = 0;
				}

				__syncthreads();

				// Loop over patterns for the neighbor
				for (int p=0; p < num_patterns; p++) {
					int pat_int = p / INT_BITS;
					int int_idx = p % INT_BITS;
					bool pat_valid = (1 << int_idx) & s_other[idx_base + pat_int];
					int base_idx = o * num_patterns * num_pat_ints + p * num_pat_ints;

					// Update mask by allowed states for this pattern
					for (int i=0; i < num_pat_ints; i++)
						s_masks[idx_base + i] |= pat_valid ? fits[base_idx + i] : 0;
					__syncthreads();
				}

				// Apply the mask to the center state
				// TODO: Check for bank conflicts here
				for (int i=0; i < num_pat_ints; i++)
					s_waves[idx_base + i] &= s_masks[idx_base + i];
			}

			// Check if state has changed and update global memory
			bool changed = false;
			for (int i=0; i < num_pat_ints; i++) {
				int bits = waves[tid_base + i];
				changed = changed || s_waves[idx_base + i] != bits;
				waves[tid_base + i] = s_waves[idx_base + i];
			}

			__syncthreads();
			workspace[tid] = changed;

			tid += blockDim.x * gridDim.x;
			tid_base = tid * num_pat_ints;
		}
	 }

	__global__
	void cudaUpdateWavesKernel(char* waves, bool* fits, int* overlays, int* workspace,
									int waves_x, int waves_y, 
									int num_patterns, int num_overlays) {
		extern __shared__ char state_allowed[];

		unsigned idx_base = threadIdx.x * num_patterns;
		unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned tid_base = tid * num_patterns;
		bool changed = false;
		
		while (tid < waves_x * waves_y) {
			// Begin by counting the existing states to determine if this tile is "locked"
			int total_states = 0;
			for (int c = 0; c < num_patterns; c++)
				total_states += waves[tid_base + c];
			bool locked = (total_states <= 1);
			//bool locked = false;

			// Individually check each pattern to see if it is allowed.
			for (int c = 0; c < num_patterns; c++) {
				int c_idx = c * num_overlays * num_patterns;
				bool start_allowed = waves[tid_base + c];
				bool allowed = start_allowed;

				// A pattern must have a valid fit on ALL neighboring sides.
				for (int o=0; o < num_overlays; o++) {
					int o_x = overlays[2 * o];
					int o_y = overlays[2 * o + 1];
					int o_idx = c_idx + o * num_patterns;
					int other_idx = tid + o_x * waves_x + o_y;
					int other_base = other_idx * num_patterns;
					bool valid = other_idx >= 0 && other_idx < waves_x * waves_y;

					// If out center is against an edge, we say the side with the
					// edge allows all states.
					bool side_allowed = locked || !valid;

					// For a given side, only 1 state is required to be a valid fit.
					for (int other_patt=0; other_patt < num_patterns; other_patt++) {
						bool waves_cond = valid && waves[other_base + other_patt];
						bool is_fit = fits[o_idx + other_patt];

						// Any fitting state will allow our center.
						side_allowed |= waves_cond && is_fit;
					}

					// All sides must be allowed.
					allowed &= side_allowed;
				}

				// A state that was once allowed has been dis-allowed
				changed |= (start_allowed != allowed);

				// Cache the new wave state into shared memory.
				state_allowed[idx_base + c] = allowed;
			}

			// Update the workspace to indicate whether this tile has changed.
			__syncthreads();
			workspace[tid] = !locked && changed;

			// Update the waves with new states from shared memory.
			__syncthreads();
			for (int c = 0; c < num_patterns; c++)
				waves[tid_base + c] = state_allowed[idx_base + c];

			tid += blockDim.x * gridDim.x;
			tid_base = tid * num_patterns;
		}
	}

	__global__
	void cudaCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
		unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

		// Set 1 if it is the intended state, 0 otherwise.
		while (tid < num_patterns) {
			waves[idx * num_patterns + tid] = tid == state;
			tid += blockDim.x * gridDim.x;
		}
	}

	__global__
	void cudaComputeEntropiesKernel(int* waves, int* entropies, int num_waves, int num_patt_ints) {
		unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
		unsigned tid_base = tid * num_patt_ints;

		// Count up the valid states in the tile, this is our entropy.
		while (tid < num_waves) {
			int total = 0;
			for (int i=0; i < num_patt_ints; i++)
				total += __popc(waves[tid_base + i]);

			__syncthreads();
			entropies[tid] = total;
			
			tid += blockDim.x * gridDim.x;
		}
	}

	__global__
	void cudaIsCollapsedKernel(int* entropies, int* workspace, int num_waves) {
		unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

		// Mark into the workspace whether a given tile has fully collapsed
		while (tid < num_waves) {
			workspace[tid] = entropies[tid] > 1;
			
			tid += blockDim.x * gridDim.x;
		}
	 }


////////////////////////////////////////////////////////////////////////////////
///																			 ///
///							Host Interface Functions							 ///
///																			 ///
////////////////////////////////////////////////////////////////////////////////

	void cudaCallReduceOrKernel(int numBlocks, int numThreads, int* data, int* output, int length) {
		// First initialize the final collection variable
		CUDA_CALL(cudaMemset(output, 0, sizeof(int)));

		// Next, kick off an efficient reduction. May loop this section to take advantage
		// of repeated reduction instead of finishing with atomic accumulation.
		switch (numThreads) {
		case 1024: cudaReduceOrKernel<1024> <<<numBlocks, numThreads>>> (data, output, length); break;
		case 512: cudaReduceOrKernel<512> <<<numBlocks, numThreads>>> (data, output, length); break;
		case 256: cudaReduceOrKernel<256> <<<numBlocks, numThreads>>> (data, output, length); break;
		case 128: cudaReduceOrKernel<128> <<<numBlocks, numThreads>>> (data, output, length); break;
		case  64: cudaReduceOrKernel< 64> <<<numBlocks, numThreads>>> (data, output, length); break;
		case  32: cudaReduceOrKernel< 32> <<<numBlocks, numThreads>>> (data, output, length); break;
		case  16: cudaReduceOrKernel< 16> <<<numBlocks, numThreads>>> (data, output, length); break;
		case   8: cudaReduceOrKernel<  8> <<<numBlocks, numThreads>>> (data, output, length); break;
		case   4: cudaReduceOrKernel<  4> <<<numBlocks, numThreads>>> (data, output, length); break;
		case   2: cudaReduceOrKernel<  2> <<<numBlocks, numThreads>>> (data, output, length); break;
		case   1: cudaReduceOrKernel<  1> <<<numBlocks, numThreads>>> (data, output, length); break;
		}
	 }

	 void cudaCallClearKernel(int* waves, int* entropies, int num_waves, int num_patterns, int num_patt_ints) {
		int numThreads = NUM_THREADS_1D(num_waves);
		int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
		cudaClearKernel<<<numBlocks, numThreads>>>(waves, entropies, num_waves, num_patterns, num_patt_ints);
	 }

	void cudaCallUpdateWavesKernel(int* waves, int* fits, int* overlays,
									int waves_x, int waves_y, 
									int num_patterns, int num_overlays, int num_pat_ints,
									int* workspace, int work_size, int* changed) {
		int total_work = waves_x * waves_y;
		int numThreads = NUM_THREADS_1D(total_work);
		int numBlocks = NUM_BLOCKS_1D(total_work, numThreads);
		int shmem_size = sizeof(char) * numThreads * num_pat_ints * 3;
		
		_cudaUpdateWavesKernel<<<numBlocks, numThreads, shmem_size>>>(
			waves, fits, overlays, workspace, 
			waves_x, waves_y, num_patterns, num_overlays, num_pat_ints);

		// The workspace now stores whether each tile was changed, reduce that into
		// a final "did the board change?"
		numThreads = NUM_THREADS_1D(work_size);
		numBlocks = NUM_BLOCKS_1D(work_size, numThreads);
		cudaCallReduceOrKernel(numBlocks, numThreads, workspace, changed, work_size);
	}

	 void cudaCallLowestEntropyKernel(int* entropies, int* workspace, int num_waves) { 
		int numThreads = NUM_THREADS_1D(num_waves);
		int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);

		// Kick off an efficient reduction. May loop this section to take advantage
		// of repeated reduction instead of finishing with atomic accumulation.
		switch (numThreads) {
		case 1024: cudaLowestEntropyKernel<1024> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case 512: cudaLowestEntropyKernel<512> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case 256: cudaLowestEntropyKernel<256> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case 128: cudaLowestEntropyKernel<128> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case  64: cudaLowestEntropyKernel< 64> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case  32: cudaLowestEntropyKernel< 32> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case  16: cudaLowestEntropyKernel< 16> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case   8: cudaLowestEntropyKernel<  8> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case   4: cudaLowestEntropyKernel<  4> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case   2: cudaLowestEntropyKernel<  2> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		case   1: cudaLowestEntropyKernel<  1> <<<numBlocks, numThreads>>> (entropies, workspace, num_waves); break;
		}

		// Kick off less efficient repeated reduction to collect the final minimum
		// and index in the workspace.
		while (numBlocks > 1) {
			numThreads = NUM_THREADS_1D(numBlocks);
			numBlocks = NUM_BLOCKS_1D(numBlocks, numThreads);
			int shmem_size = sizeof(int) * 2 * numThreads;
			cudaReduceMinIdxKernel<<<numBlocks, numThreads, shmem_size>>>(workspace, numBlocks);
		}
	 }

	void cudaCallCollapseWaveKernel(char* waves, int idx, int state, int num_patterns) {
		// NOTE: Does not update entropy value, only updates waves
		int numThreads = NUM_THREADS_1D(num_patterns);
		int numBlocks = NUM_BLOCKS_1D(num_patterns, numThreads);
		cudaCollapseWaveKernel<<<numBlocks, numThreads>>>(waves, idx, state, num_patterns);
	}

	void cudaCallComputeEntropiesKernel(int* waves, int* entropies, int num_waves, int num_patt_ints) {
		int numThreads = NUM_THREADS_1D(num_waves);
		int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
		cudaComputeEntropiesKernel<<<numBlocks, numThreads>>>(waves, entropies, num_waves, num_patt_ints);
	}

	void cudaCallIsCollapsedKernel(int* entropies, int* workspace, int* is_collapsed, int num_waves) {
		int numThreads = NUM_THREADS_1D(num_waves);
		int numBlocks = NUM_BLOCKS_1D(num_waves, numThreads);
		cudaIsCollapsedKernel<<<numBlocks, numThreads>>>(entropies, workspace, num_waves);

		// Workspace stores whether a tile is NOT collapsed. Reduce that into a final
		// "is there any non-collapsed tile"
		cudaCallReduceOrKernel(numBlocks, numThreads, workspace, is_collapsed, num_waves);
	 }

}
