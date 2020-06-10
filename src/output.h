#pragma once
#include "model.h"

namespace wfc
{
	/**
	 * \brief Renders the board state of the given model into an output image. Patterns
	 * must be ordered the same way as it's counts are passed into the model.
	 */
	void render_image(Model& model, std::vector<cv::Mat>& patterns, cv::Mat& out_img);

	void show_patterns(Model& model, std::vector<cv::Mat>& patterns, 
		std::vector<Pair>& overlays, std::vector<std::vector<int>>& fit_table);
}
