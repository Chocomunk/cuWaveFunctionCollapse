#include "model.h"
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "cuwfc.cuh"

#include <chrono>

namespace wfc
{

////////////////////////////////////////////////////////////////////////////////
///																			 ///
///									CPU Model								 ///
///																			 ///
////////////////////////////////////////////////////////////////////////////////

	CpuModel::CpuModel(Pair& output_shape, const int num_patterns, const int overlay_count,
		const char dim, const bool periodic, const int iteration_limit) :
		Model(output_shape, num_patterns, overlay_count, dim, periodic, iteration_limit),
		num_patt_2d_(num_patterns, num_patterns) {
		srand(time(nullptr));

		// Initialize collections.
		propagate_stack_ = std::vector<Waveform>(wave_shape.size * num_patterns);
		entropy_ = std::vector<int>(wave_shape.size);
		waves_ = std::vector<char>(wave_shape.size * num_patterns);
		observed_ = std::vector<int>(wave_shape.size);
		compatible_neighbors_ = std::vector<int>(wave_shape.size * num_patterns * overlay_count);

		std::cout << "Patterns: " << num_patterns << std::endl;
		std::cout << "Overlay Count: " << overlay_count << std::endl;
		std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
	}

	void CpuModel::generate(std::vector<Pair>& overlays, std::vector<int>& counts, std::vector<std::vector<int>>& fit_table) {
		std::cout << "Called Generate" << std::endl;

		// Initialize board into complete superposition, and pick a random wave to collapse
		clear(fit_table);
		Pair lowest_entropy_idx = Pair(rand_int(wave_shape.x), rand_int(wave_shape.y));
		int iteration = 0;
		while ((iteration_limit < 0 || iteration < iteration_limit) && lowest_entropy_idx.non_negative()) {
			/* Standard wfc Loop:
			 *		1. Observe a wave and collapse it's state
			 *		2. Propagate the changes throughout the board and update superposition
			 *		   to only allowed states
			 *		3. After the board state has stabilized, find the position of
			 *		   lowest entropy (most likely to be observed) for the next
			 *		   observation.
			 */
			observe_wave(lowest_entropy_idx, counts);
			propagate(overlays, fit_table);
			get_lowest_entropy(lowest_entropy_idx);

			iteration += 1;
			if (iteration % 1000 == 0)
				std::cout << "iteration: " << iteration << std::endl;
		}
		std::cout << "Finished Algorithm in " << iteration << " iterations" << std::endl;
	}

	void CpuModel::get_superposition(const int row, const int col, std::vector<int>& patt_idxs) {
		const int idx_row_col_patt_base = row * wave_shape.x * num_patterns + col * num_patterns;

		// Determines the superposition of patterns at this position.
		int num_valid_patterns = 0;
		for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
			if (waves_[patt_idx + idx_row_col_patt_base]) {
				num_valid_patterns += 1;
				patt_idxs.push_back(patt_idx);
			}
		}
	}

	void CpuModel::clear(std::vector<std::vector<int>>& fit_table) {
		for (int wave = 0; wave < wave_shape.size; wave++) {
			for (int patt = 0; patt < num_patterns; patt++) {
				waves_[wave * num_patterns + patt] = true;
				for (int overlay = 0; overlay < overlay_count; overlay++) {
					// Reset count of compatible neighbors in the fit table (to all states)
					compatible_neighbors_[(wave * num_patterns + patt) * overlay_count + overlay] = fit_table[patt * overlay_count + (overlay + 2) % overlay_count].size();
				}
			}
			observed_[wave] = -1;
			entropy_[wave] = num_patterns;
		}
	}

	void CpuModel::get_lowest_entropy(Pair& idx) {
		int r = -1; int c = -1;
		int lowest_entropy = -1;
		const int waves_count = wave_shape.size;

		// Checks all non-collapsed positions to find the position of lowest entropy.
		for (int wave_idx = 1; wave_idx < waves_count; wave_idx++) {
			const int entropy_val = entropy_[wave_idx];
			if ((lowest_entropy < 0 || entropy_val < lowest_entropy) && entropy_val > 0 && observed_[wave_idx] == -1) {
				lowest_entropy = entropy_val;
				r = wave_idx / wave_shape.x; c = wave_idx % wave_shape.x;
			}
		}
		idx.x = c;; idx.y = r;
	}

	void CpuModel::observe_wave(Pair& pos, std::vector<int>& counts) {
		const int idx_row_col_patt_base = get_idx(pos, wave_shape, num_patterns, 0);

		// Determines superposition of states and their total frequency counts.
		int possible_patterns_sum = 0;
		std::vector<int> possible_indices;
		for (int i = 0; i < num_patterns; i++) {
			if (waves_[idx_row_col_patt_base + i]) {
				possible_patterns_sum += counts[i];
				possible_indices.push_back(i);
			}
		}

		int rnd = rand_int(possible_patterns_sum) + 1;
		int collapsed_index;

		// Randomly selects a state for collapse. Weighted by state frequency count.
		for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
			if (waves_[collapsed_index + idx_row_col_patt_base]) {
				rnd -= counts[collapsed_index];
			}
		}
		collapsed_index -= 1;	// Counter-action against additional increment from for-loop

		// Bans all other states, since we have collapsed to a single state.
		for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
			if (waves_[patt_idx + idx_row_col_patt_base] != (patt_idx == collapsed_index))
				ban_waveform(Waveform(pos, patt_idx));
		}

		// Assigns the final state of this position.
		observed_[get_idx(pos, wave_shape, 1, 0)] = collapsed_index;
	}

	void CpuModel::propagate(std::vector<Pair>& overlays, std::vector<std::vector<int>>& fit_table) {
		// int iterations = 0;
		while (stack_index_ > 0) {
			const Waveform wave_f = pop_waveform();
			Pair wave = wave_f.pos;
			const int pattern_i = wave_f.state;

			// Check all overlayed tiles.
			for (int overlay = 0; overlay < overlay_count; overlay++) {
				Pair wave_o;

				// If periodic, wrap positions past the edge of the board.
				if (periodic) {
					wave_o = (wave + overlays[overlay]) % wave_shape;
				}
				else {
					wave_o = wave + overlays[overlay];
				}

				const int wave_o_i = get_idx(wave_o, wave_shape, num_patterns, 0);
				const int wave_o_i_base = get_idx(wave_o, wave_shape, 1, 0);

				// If position is valid and non-collapsed, then propagate changes through
				// this position (wave_o).
				if (wave_o.non_negative() && wave_o < wave_shape && entropy_[wave_o_i_base] > 1) {
					auto valid_patterns = fit_table[pattern_i * overlay_count + overlay];
					for (int pattern_2 : valid_patterns) {
						if (waves_[wave_o_i + pattern_2]) {
							// Get opposite overlay
							const int compat_idx = (wave_o_i + pattern_2) * overlay_count + overlay;
							compatible_neighbors_[compat_idx]--;

							// If there are no valid neighbors left, this state is impossible.
							if (compatible_neighbors_[compat_idx] == 0)
								ban_waveform(Waveform(wave_o, pattern_2));
						}
					}
				}
			}

			// iterations += 1;
			// if (iterations % 1000 == 0)
			// 	std::cout << "propagation iteration: " << iterations << ", propagation stack size: " << stack_index_ << std::endl;
		}
	}

	void CpuModel::stack_waveform(Waveform& wave) {
		propagate_stack_[stack_index_] = wave;
		stack_index_ += 1;
	}

	Waveform CpuModel::pop_waveform() {
		stack_index_ -= 1;
		return propagate_stack_[stack_index_];
	}

	void CpuModel::ban_waveform(Waveform wave) {
		const int waves_idx = get_idx(wave.pos, wave_shape, num_patterns, wave.state);
		const int wave_i = get_idx(wave.pos, wave_shape, 1, 0);

		// Mark this specific Waveform as disallowed, and update neighboring patterns
		// to block propagation through this state.
		waves_[waves_idx] = false;
		for (int overlay = 0; overlay < overlay_count; overlay++) {
			compatible_neighbors_[waves_idx * overlay_count + overlay] = 0;
		}
		stack_waveform(wave);	// Propagate changes through neighboring positions.

		entropy_[wave_i] -= 1;
	}


////////////////////////////////////////////////////////////////////////////////
///																			 ///
///									GPU Model								 ///
///																			 ///
////////////////////////////////////////////////////////////////////////////////

	GpuModel::GpuModel(Pair& output_shape, const int num_patterns, const int overlay_count,
		const char dim, const bool periodic, const int iteration_limit) :
		Model(output_shape, num_patterns, overlay_count, dim, periodic, iteration_limit) {
		srand(time(nullptr));

		waves_padded_length_ = next_pow_2(wave_shape.size);

		// Allocate collections.
		CUDA_CALL(cudaMalloc((void**)&dev_entropy_, sizeof(int) * waves_padded_length_));
		CUDA_CALL(cudaMalloc((void**)&dev_waves_, sizeof(char) * wave_shape.size * num_patterns));
		CUDA_CALL(cudaMalloc((void**)&dev_workspace_, sizeof(int) * waves_padded_length_));
		CUDA_CALL(cudaMalloc((void**)&dev_changed_, sizeof(int)));
		CUDA_CALL(cudaMalloc((void**)&dev_is_collapsed_, sizeof(int)));

		host_waves_ = new char[wave_shape.size * num_patterns];
		host_single_wave_ = new char[num_patterns];
		host_changed_ = new int;
		host_lowest_entropy = new int;
		host_is_collapsed = new int;

		// Some necessary initializations
		CUDA_CALL(cudaMemset(dev_entropy_, 0, sizeof(int) * waves_padded_length_));
		CUDA_CALL(cudaMemset(dev_workspace_, 0, sizeof(int) * waves_padded_length_));

		*host_is_collapsed = false;

		std::cout << "Patterns: " << num_patterns << std::endl;
		std::cout << "Overlay Count: " << overlay_count << std::endl;
		std::cout << "Wave Shape: " << wave_shape.y << " x " << wave_shape.x << std::endl;
	}

	GpuModel::~GpuModel() {
		CUDA_CALL(cudaFree(dev_entropy_));
		CUDA_CALL(cudaFree(dev_waves_));
		CUDA_CALL(cudaFree(dev_workspace_));
		CUDA_CALL(cudaFree(dev_changed_));
		CUDA_CALL(cudaFree(dev_is_collapsed_));
		delete[] host_waves_;
		delete[] host_single_wave_;
		delete host_changed_;
		delete host_lowest_entropy;
		delete host_is_collapsed;
	}


	void GpuModel::generate(std::vector<Pair>& overlays, std::vector<int>& counts, std::vector<std::vector<int>>& fit_table) {
		std::cout << "Called Generate" << std::endl;
		// Convert fit_table and overlays to GPU data for propagate kernel
		bool* dev_fit_table = get_device_fit_table(fit_table);
		int* dev_overlays = get_device_overlays(overlays);

		// Initialize board into complete superposition, and pick a random wave to collapse
		clear(fit_table);
		*host_lowest_entropy = rand_int(wave_shape.size);

		int iteration = 0;
		while ((iteration_limit < 0 || iteration < iteration_limit) && *host_lowest_entropy < wave_shape.size) {
			/* Standard wfc Loop:
			 *		1. Observe a wave and collapse it's state
			 *		2. Propagate the changes throughout the board and update superposition
			 *		   to only allowed states
			 *		3. After the board state has stabilized, find the position of
			 *		   lowest entropy (most likely to be observed) for the next
			 *		   observation.
			 */
			// TODO: Pass in proper converted GPU data

			auto t1 = std::chrono::high_resolution_clock::now();
			observe_wave(*host_lowest_entropy, counts);	// Collapse a wave
			auto t2 = std::chrono::high_resolution_clock::now();
			propagate(dev_overlays, dev_fit_table);			// Propagate through all waves
			auto t3 = std::chrono::high_resolution_clock::now();
			update_entropies();								// Assign entropies
			auto t4 = std::chrono::high_resolution_clock::now();
			//check_completed();								// Reduce into host_is_collapsed
			get_lowest_entropy();							// Reduce into host_lowest_entropy
			auto t5 = std::chrono::high_resolution_clock::now();

			obs_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			pro_time += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
			upd_time += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
			low_time += std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();

			iteration += 1;
			if (iteration % 1000 == 0)
				std::cout << "iteration: " << iteration << std::endl;
		}

		CUDA_CALL(cudaFree(dev_fit_table));
		CUDA_CALL(cudaFree(dev_overlays));

		// Update the host array with the device array
		apply_host_waves();
		std::cout << "Finished Algorithm in " << iteration << " iterations" << std::endl;
		std::cout << "Total observation time: " << obs_time << std::endl;
		std::cout << "Total propagation time: " << pro_time << std::endl;
		std::cout << "Total update time: " << upd_time << std::endl;
		std::cout << "Total lowest_entropy time: " << low_time << std::endl;
		std::cout << "Total propagation call time: " << prop_call_time << std::endl;
		std::cout << "Total propagation copy time: " << prop_copy_time << std::endl;
	}

	void GpuModel::get_superposition(const int row, const int col, std::vector<int>& patt_idxs) {
		const int idx_row_col_patt_base = row * wave_shape.x * num_patterns + col * num_patterns;
		// Determines the superposition of patterns at this position.
		int num_valid_patterns = 0;
		for (int patt_idx = 0; patt_idx < num_patterns; patt_idx++) {
			if (host_waves_[patt_idx + idx_row_col_patt_base]) {
				num_valid_patterns += 1;
				patt_idxs.push_back(patt_idx);
			}
		}
	}

	void GpuModel::clear(std::vector<std::vector<int>>& fit_table) {
		cudaCallClearKernel(dev_waves_, dev_entropy_, wave_shape.size, num_patterns);
	}

	void GpuModel::get_lowest_entropy() {
		cudaCallLowestEntropyKernel(dev_entropy_, dev_workspace_, waves_padded_length_);
		CUDA_CALL(cudaMemcpy(host_lowest_entropy, dev_workspace_ + 1, sizeof(int), cudaMemcpyDeviceToHost));
	}

	void GpuModel::observe_wave(int idx, std::vector<int>& counts) {
		const int idx_base = idx * num_patterns;

		CUDA_CALL(cudaMemcpy(host_single_wave_, (dev_waves_ + idx_base), 
			sizeof(char) * num_patterns, cudaMemcpyDeviceToHost));

		// Determines superposition of states and their total frequency counts.
		int possible_patterns_sum = 0;
		for (int i = 0; i < num_patterns; i++) {
			if (host_single_wave_[i]) {
				possible_patterns_sum += counts[i];
			}
		}

		int rnd = rand_int(possible_patterns_sum) + 1;
		int collapsed_index;

		// Randomly selects a state for collapse. Weighted by state frequency count.
		for (collapsed_index = 0; collapsed_index < num_patterns && rnd>0; collapsed_index++) {
			if (host_single_wave_[collapsed_index]) {
				rnd -= counts[collapsed_index];
			}
		}
		collapsed_index -= 1;	// Counter-action against additional increment from for-loop
		assert(collapsed_index >= 0 && collapsed_index < num_patterns);

		// Update wave in the GPU data
		cudaCallCollapseWaveKernel(dev_waves_, idx,collapsed_index, num_patterns);
	}

	void GpuModel::propagate(int* overlays, bool* fit_table) {
		// Continue updating waves in GPU as long as there are changes
		bool changed = true;
		while (changed) {
			auto t1 = std::chrono::high_resolution_clock::now();
			cudaCallUpdateWavesKernel(dev_waves_, fit_table, overlays,
				wave_shape.x, wave_shape.y, num_patterns, overlay_count,
				dev_workspace_, waves_padded_length_, dev_changed_);
			auto t2 = std::chrono::high_resolution_clock::now();
			cudaDeviceSynchronize();

			// Determine whether there were changes from reduction output
			CUDA_CALL(cudaMemcpy(host_changed_, dev_changed_, sizeof(int), cudaMemcpyDeviceToHost));
			changed = *host_changed_ > 0;
			auto t3 = std::chrono::high_resolution_clock::now();

			prop_call_time += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
			prop_copy_time += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

		}
	}

	bool* GpuModel::get_device_fit_table(std::vector<std::vector<int>>& fit_table) const {
		int length = num_patterns * overlay_count * num_patterns;
		
		// Rebuild fit_table from adjacency list to edge-matrix in CPU first
		bool* host_fit_table = new bool[length];
		memset(host_fit_table, 0, sizeof(bool) * length);
		for (int c=0; c < num_patterns; c++) {
			for (int o=0; o < overlay_count; o++) {
				int base_idx = c * overlay_count + o;
				auto valid_patterns = fit_table[base_idx];
				for (int other_patt: valid_patterns) {
					host_fit_table[base_idx * num_patterns + other_patt] = true;
				}
			}
		}

		// Transfer new fit_table to GPU device
		bool* dev_fit_table;
		CUDA_CALL(cudaMalloc((void**)&dev_fit_table, sizeof(bool) * length));
		CUDA_CALL(cudaMemcpy(dev_fit_table, host_fit_table, sizeof(bool) * length, cudaMemcpyHostToDevice));

		// Free CPU matrix and return device pointer
		delete[] host_fit_table;
		return dev_fit_table;
	}

	int* GpuModel::get_device_overlays(std::vector<Pair> &overlays) const {
		// Returns a "pair" array where paired ints are sequential
		// Because of the extra "size" parameter that we dont want, make a CPU
		// array to copy to GPU. May remove "size" parameter in the future.
		int* host_overlays = new int[overlay_count * 2];
		for (int i=0; i < overlay_count; i++) {
			host_overlays[2 * i] = overlays[i].x;
			host_overlays[2 * i + 1] = overlays[i].y;
		}

		// Transfer host overlays to GPU device
		int* dev_overlays;
		CUDA_CALL(cudaMalloc((void**)&dev_overlays, sizeof(int) * 2 * overlay_count));
		CUDA_CALL(cudaMemcpy(dev_overlays, host_overlays, 
				sizeof(int) * 2 * overlay_count, cudaMemcpyHostToDevice));

		// Free CPU array and return device pointer
		delete[] host_overlays;
		return dev_overlays;
	}

	void GpuModel::apply_host_waves() const {
		// Copy GPU waves to CPU
		CUDA_CALL(cudaMemcpy(host_waves_, dev_waves_,
			sizeof(char) * wave_shape.size * num_patterns,
			cudaMemcpyDeviceToHost));
	}

	void GpuModel::update_entropies() const {
		cudaCallComputeEntropiesKernel(dev_waves_, dev_entropy_, wave_shape.size, num_patterns);
	}

	void GpuModel::check_completed() const {
		// Call check and reduction kernel then determine from the reduction output
		cudaCallIsCollapsedKernel(dev_entropy_, dev_workspace_, dev_is_collapsed_, waves_padded_length_);
		CUDA_CALL(cudaMemcpy(host_is_collapsed, dev_is_collapsed_, sizeof(int), cudaMemcpyDeviceToHost));
		*host_is_collapsed = (*host_is_collapsed == 0);
	}

	void GpuModel::print_waves() const {
		// Get our waves into CPU array first. This transaction is expensive, so
		// Don't use this in the final version.
		apply_host_waves();
		for (int r = 0; r < wave_shape.y; r++) {
			int r_b = r * wave_shape.x * num_patterns;
			std::cout << "[ ";
			for (int c = 0; c < wave_shape.x; c++) {
				int r_c = r_b + c * num_patterns;
				std::cout << "< ";
				for (int p=0; p < num_patterns; p++) {
					std::cout << (int)host_waves_[r_c + p] << " ";
				}
				std::cout << ">, ";
			}
			std::cout << "], " << std::endl;
		}
	}

	void GpuModel::print_entropy() const {
		// This normally isn't supposed to be on CPU, so we have to allocate a
		// new array just to accomodate it. Don't use this in final version
		int* h_e = new int[waves_padded_length_];
		CUDA_CALL(cudaMemcpy(h_e, dev_entropy_, sizeof(int) * waves_padded_length_, cudaMemcpyDeviceToHost));
		
		for (int r = 0; r < wave_shape.y; r++) {
			int r_b = r * wave_shape.x;
			std::cout << "[ ";
			for (int c = 0; c < wave_shape.x; c++) {
				std::cout << h_e[r_b + c] << " ";
			}
			std::cout << "], " << std::endl;
		}
		std::cout << std::endl;

		delete[] h_e;
	}

	void GpuModel::print_workspace() const {
		// This normally isn't supposed to be on CPU, so we have to allocate a
		// new array just to accomodate it. Don't use this in final version.
		int* h_e = new int[waves_padded_length_];
		CUDA_CALL(cudaMemcpy(h_e, dev_workspace_, sizeof(int) * waves_padded_length_, cudaMemcpyDeviceToHost));

		std::cout << "< ";
		for (int i=0; i < waves_padded_length_; i++) {
			std::cout << h_e[i] << " ";
		}
		std::cout << ">" << std::endl;

		delete[] h_e;
	}
}
