#pragma once

// TODO: Complete function definitions and implement their kernels in the cu file

// NOTE: These functions ARE NOT actualy cuda kernels, they are a "middle-man"
//		 that calls the real kernels that will be defined in cuwfc.cu

void cudaClearKernel();

void cudaUpdateWavesKernel(char* waves, int waves_x, int waves_y, int num_patterns);

void cudaLowestEntropyKernel();

void cudaCollapseWaveKernel(char* waves, int idx, int state, 
							int waves_x, int waves_y, int num_patterns);

void cudaComputeEntropiesKernel(char* waves, int* entropies, 
								int waves_x, int waves_y, int num_patterns);
