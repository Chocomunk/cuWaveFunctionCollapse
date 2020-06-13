#pragma once
#include <vector>
#include "wfc_util.h"

/* Dimension legend
	Template counts: T
	Pattern counts: N
	Tile/Pattern dim: D
	Overlay counts: O
	Wave shape x: WX
	Wave shape y: WY
*/

namespace wfc
{
	class Model {
	public:
		const char dim;
		const int iteration_limit;
		const int num_patterns;
		const int overlay_count;
		const bool periodic;
		Pair wave_shape;

		virtual ~Model() = default;

		Model(Pair& output_shape, const int num_patterns, const int overlay_count,
			const char dim, const bool periodic = false, const int iteration_limit = -1)
				: dim(dim), iteration_limit(iteration_limit), num_patterns(num_patterns),
				overlay_count(overlay_count),	periodic(periodic),
				wave_shape(output_shape.x + 1 - dim, output_shape.y + 1 - dim) {}

		/**
		 * \brief Runs the wfc algorithm and stores the output image (call 'get_image'
		 * to access).
		 */
		virtual void generate(std::vector<Pair> &overlays, std::vector<int> &counts, std::vector<std::vector<int>> &fit_table) = 0;
		
		/**
		 * \brief Generates an image of the superpositions of the wave at (row, col),
		 * and stores the result in the output image.
		 */
		virtual void get_superposition(int row, int col, std::vector<int> &patt_idxs) = 0;
		
		/**
		 * \brief Resets all tiles to a perfect superposition.
		 */
		virtual void clear(std::vector<std::vector<int>> &fit_table) = 0;
	};

	class CpuModel : public Model {

	public:
		/**
		 * \brief Initializes a model instance and allocates all workspace data.
		 */
		CpuModel(Pair &output_shape, const int num_patterns, const int overlay_count, 
			const char dim, const bool periodic=false, const int iteration_limit=-1);
		
		/**
		 * \brief Runs the wfc algorithm and stores the output image (call 'get_image'
		 * to access).
		 */
		void generate(std::vector<Pair> &overlays, std::vector<int> &counts, std::vector<std::vector<int>> &fit_table) override;
		
		/**
		 * \brief Generates an image of the superpositions of the wave at (row, col),
		 * and stores the result in the output image.
		 */
		void get_superposition(int row, int col, std::vector<int> &patt_idxs) override;
		
		/**
		 * \brief Resets all tiles to a perfect superposition.
		 */
		void clear(std::vector<std::vector<int>> &fit_table) override;

	private:
		Pair num_patt_2d_;
		int stack_index_ = 0;

		/**
		 * \brief Fixed-space workspace stack for propagation step.
		 *
		 * Shape: [WX * WY * N]
		 */
		std::vector<Waveform> propagate_stack_;

		/**
		 * \brief Stores the entropy (number of valid patterns) for a given position.
		 *
		 * Shape: [WX, WY]
		 */
		std::vector<int> entropy_;

		/**
		 * \brief Stores whether a specific Waveform (position, state) is allowed
		 * (true/false).
		 *
		 * Shape: [WX, WY, N]
		 */
		std::vector<char> waves_;

		/**
		 * \brief Stores the index of the final collapsed pattern for a given position.
		 *
		 * Shape: [WX, WY]
		 */
		std::vector<int> observed_;

		/**
		 * \brief Stores a count of the number of compatible neighbors for this pattern.
		 * If there are no compatible neighbors, then it is impossible for this pattern
		 * to occur and we should ban it.
		 *
		 * Shape: [WX, WY, N, O]
		 */
		std::vector<int> compatible_neighbors_;
		
		/**
		 * \brief Finds the wave with lowest entropy and stores it's position in idx
		 */
		void get_lowest_entropy(Pair &idx);
		
		/**
		 * \brief Performs an observation on the wave at the given position and
		 * collapses it to a single state.
		 */
		void observe_wave(Pair &pos, std::vector<int> &counts);
		
		/**
		 * \brief Iteratively collapses waves in the tilemap until no conflicts exist.
		 * Meant to be used after collapsing a wave by observing it.
		 */
		void propagate(std::vector<Pair>& overlays, std::vector<std::vector<int>> &fit_table);

		/**
		 * \brief Adds a wave to the propagation stack to propagate changes to it's
		 * neighbors (determined by overlays).
		 */
		void stack_waveform(Waveform& wave);
		
		/**
		 * \return The last Waveform added to the propagation stack.
		 */
		Waveform pop_waveform();

		/**
		 * \brief Bans a specific state at the Waveform position. Partially collapses
		 * the overall state at that position and reduces it's entropy.
		 */
		void ban_waveform(Waveform wave);
	};

	class GpuModel : public Model {
	public:
		/**
		 * \brief Initializes a model instance and allocates all workspace data.
		 */
		GpuModel(Pair &output_shape, const int num_patterns, const int overlay_count, 
			const char dim, const bool periodic=false, const int iteration_limit=-1);

		/**
		 * \brief Cleans up all dynamically allocated GPU and CPU arrays
		 */
		~GpuModel();
		
		/**
		 * \brief Runs the wfc algorithm and stores the output image (call 'get_image'
		 * to access).
		 */
		void generate(std::vector<Pair> &overlays, std::vector<int> &counts, std::vector<std::vector<int>> &fit_table) override;
		
		/**
		 * \brief Generates an image of the superpositions of the wave at (row, col),
		 * and stores the result in the output image.
		 */
		void get_superposition(int row, int col, std::vector<int> &patt_idxs) override;
		
		/**
		 * \brief Resets all tiles to a perfect superposition.
		 */
		void clear(std::vector<std::vector<int>> &fit_table) override;

		// Performance Analysis
		int obs_time = 0;
		int pro_time = 0;
		int upd_time = 0;
		int low_time = 0;

		int prop_call_time = 0;
		int prop_copy_time = 0;

		int obs_copy_time = 0;
		int obs_cpu_time = 0;
		int obs_gpu_time = 0;

	private:
		int* dev_entropy_;
		char* dev_waves_;
		int* dev_workspace_;
		int* dev_changed_;
		int* dev_is_collapsed_;

		char* host_waves_;
		char* host_single_wave_;
		int* host_changed_;
		int* host_lowest_entropy;
		int* host_is_collapsed;

		int waves_padded_length_;
		
		/**
		 * \brief Finds the wave with lowest entropy and stores it's position in idx
		 */
		void get_lowest_entropy();
		
		/**
		 * \brief Performs an observation on the wave at the given position and
		 * collapses it to a single state.
		 */
		void observe_wave(int idx, std::vector<int> &counts);
		
		/**
		 * \brief Iteratively collapses waves in the tilemap until no conflicts exist.
		 * Meant to be used after collapsing a wave by observing it.
		 */
		void propagate(int* overlays, bool* fit_table);

		/**
		 * \brief Moves a given fit_table to GPU device memory
		 * \return The pointer to the GPU fit_table
		 */
		bool* get_device_fit_table(std::vector<std::vector<int>>& fit_table) const;

		/**
		 * \brief Moves a given overlay set to GPU device memory
		 * \return The pointer to the GPU overlay set
		 */
		int* get_device_overlays(std::vector<Pair> &overlays) const;

		/**
		 * \brief Moves generated waves from GPU array to CPU array
		 */
		void apply_host_waves() const;

		/**
		 * \brief Updates the device entropy array with the device waves array.
		 * Each entropy element should be a count of the valid patterns in the
		 * corresponding element of the waves array.
		 */
		void update_entropies() const;

		/**
		 * \brief Does a reduction over the entropies to see if the wavefunction
		 * has fully collapsed (all entropies are 1)
		 */
		void check_completed() const;

		/**
		 * \brief Copies the device waves array to CPU and prints it legibly.
		 * For debugging purposes only.
		 */
		void print_waves() const;

		/**
		 * \brief Copies the device entropy array to CPU and prints it legibly
		 * For debugging purposes only.
		 */
		void print_entropy() const;

		/**
		 * \brief Copies the device workspace array to CPU and prints it legibly
		 * For debugging purposes only.
		 */
		void print_workspace() const;
	};
}
