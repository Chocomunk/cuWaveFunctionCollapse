#include "output.h"
#include "input.h"

namespace wfc
{
	void render_image(Model& model, std::vector<cv::Mat>& patterns, cv::Mat &out_img) {
		const int height = model.wave_shape.y;
		const int width = model.wave_shape.x;
		const int dim = model.dim;
		std::vector<int> valid_patts = std::vector<int>(model.num_patterns);
		
		for (int row=0; row < height; row++) {
			for (int col=0; col < width; col++) {
				// Get the valid patterns for this position and render the
				// average of each pattern
				valid_patts.clear();
				model.get_superposition(row, col, valid_patts);
				for (int r = row; r < row + dim; r++) {
					for (int c = col; c < col + dim; c++) {
						BGR& bgr = out_img.ptr<BGR>(r)[c];
						if (valid_patts.empty()) {
							// Error: No valid patterns (magenta). Generating a board that is
							// guaranteed to have no errors is NP.
							bgr = BGR(204, 51, 255);
						} else {
							bgr = BGR(0, 0, 0);
							for (int patt_idx: valid_patts)
								bgr += patterns[patt_idx].ptr<BGR>(r - row)[c - col] / valid_patts.size();
						}
					}
				}
			}
		}

		std::cout << "Finished Rendering" << std::endl;
	}

	void show_patterns(Model& model, std::vector<cv::Mat>& patterns, 
		std::vector<Pair>& overlays, std::vector<std::vector<int>>& fit_table) {
		std::cout << "Pattern Viewer Controls:" << std::endl <<
			"\tEsc: Quit and continue" << std::endl <<
			"\tM: Next left pattern" << std::endl <<
			"\t(Any Other Key): Next right pattern" << std::endl;

		bool break_all = false;
		std::cout << std::endl;

		for (int pat_idx1 = 0; pat_idx1 < model.num_patterns; pat_idx1++) {
			cv::Mat scaled1;
			auto patt1 = patterns[pat_idx1];
			cv::resize(patt1, scaled1, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
			std::cout << "Pattern: " << pat_idx1 << " | Pattern count: " << patterns[pat_idx1] << std::endl;

			for (int overlay_idx = 0; overlay_idx < model.overlay_count; overlay_idx++) {
				size_t index = pat_idx1 * model.overlay_count + overlay_idx;
				auto valid_patterns = fit_table[index];
				Pair overlay = overlays[overlay_idx];
				Pair opposite = overlays[(overlay_idx + 2) % model.overlay_count];
				std::cout << "Valid Patterns: ";
				for (int pattern_2 : valid_patterns) std::cout << pattern_2 << ", ";
				std::cout << std::endl;
				std::cout << "Overlay: " << overlay << " | Opposite: " << opposite << std::endl;

				bool break_part = false;

				for (int pat_idx2 = 0; pat_idx2 < model.num_patterns; pat_idx2++) {
					auto patt2 = patterns[pat_idx2];
					cv::Mat scaled2;
					cv::resize(patt2, scaled2, cv::Size(128, 128), 0.0, 0.0, cv::INTER_AREA);
					cv::Mat comb;
					cv::hconcat(scaled1, scaled2, comb);
					std::cout << "template: " << pat_idx1 << ", conv: " << pat_idx2 << ", result:" << std::endl;

					std::cout << "    " << "Table: " << (std::find(valid_patterns.begin(), valid_patterns.end(),
					                                               pat_idx2) != valid_patterns.end()) << " | Calc: "
						<< overlay_fit(patt1, patt2, overlay, model.dim) << std::endl;

					std::cout << "    " << patterns_equal(patt1, patt2) << ": " << "Patterns are equal" << std::endl;

					cv::imshow("comparison", comb);
					int k = cv::waitKey(0);
					if (k == 27) {
						break_all = true;
						break_part = true;
						break;
					}
					else if (k == int('m')) {
						break_part = true;
						break;
					}
					// else if (k == int('n')) break;
				}
				std::cout << std::endl;
				if (break_all || break_part) break;
			}
			if (break_all) break;
		}

	}
}
