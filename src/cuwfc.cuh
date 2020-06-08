#pragma once

// TODO: Complete function definitions and implement their kernels in the cu file

// NOTE: These functions ARE NOT actualy cuda kernels, they are a "middle-man"
//		 that calls the real kernels that will be defined in cuwfc.cu

void cudaClearKernel();

void cudaUpdateWavesKernel();

void cudaLowestEntropyKernel();

void cudaCollapseWaveKernel();

