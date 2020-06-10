#pragma once

// TODO: Complete function definitions and implement their kernels in the cu file

// NOTE: These functions ARE NOT actually cuda kernels, they are a "middle-man"
//		 which calls the real kernels that will be defined in cuwfc.cu

void cudaCallClearKernel(char* waves, int* entropies, int num_waves, int num_patterns);

void cudaCallUpdateWavesKernel(char* waves, bool* fits, int* overlays,
								int waves_x, int waves_y, 
								int num_patterns, int num_overlays,
								bool* changes, int changes_size, int* changed);

void cudaCallLowestEntropyKernel(int* entropies, int num_waves, int num_patterns, int* lowest_entropy_idx);

void cudaCallCollapseWaveKernel(char* waves, int idx, int state, int num_patterns);

void cudaCallComputeEntropiesKernel(char* waves, int* entropies, int num_waves, int num_patterns);
