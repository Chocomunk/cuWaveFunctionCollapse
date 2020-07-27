#pragma once

#define INT_BITS (32)

namespace wfc
{
	/**
	 * \brief Resets all waves to perfect superpositions and updates the entropy accordingly.
	 */
	void cudaCallClearKernel(int* waves, int* entropies, int num_waves, int num_patterns, int num_patt_ints);

	/**
	 * \brief Update each tile's states for 1 iteration. Run an or-reduce to determine
	 * if any tile has changed.
	 */
	void cudaCallUpdateWavesKernel(int* waves, int* fits, int* overlays,
									int waves_x, int waves_y, 
									int num_patterns, int num_overlays, int num_pat_ints,
									int* workspace, int work_size, int* changed);

	/**
	 * \brief Run a reduction to extract the index of the lowest entropy tile.
	 */
	void cudaCallLowestEntropyKernel(int* entropies, int* workspace, int num_waves);

	/**
	 * \brief Collapse one tile into a specific state.
	 */
	void cudaCallCollapseWaveKernel(char* waves, int idx, int state, int num_patterns);

	/**
	 * \brief Count the number of valid states in each tile, this is the entropy for
	 * that tile.
	 */
	void cudaCallComputeEntropiesKernel(int* waves, int* entropies, int num_waves, int num_patt_ints);

	/**
	 * \brief Run an or-reduction to determine if any tile is not fully collapsed.
	 */
	void cudaCallIsCollapsedKernel(int* entropies, int* workspace, int* is_collapsed, int num_waves);
}
