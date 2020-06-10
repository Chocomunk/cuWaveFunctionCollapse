#include "input.h"
#include "wfc.h"
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace wfc;

int main(int argc, char** argv) {
	char* tiles_dir;
	int tile_dim = 3;
	int rotate = 1;
	int periodic = 1;
	int width = 64;
	int height = 64;
	int render = 0;
	char* out_name;

	if (!(argc > 2)) {
		std::cout << "Usage {arg_name (options) | default}:" << std::endl <<
			"\twfc {image folder} {tile dim | 3} {rotate? (0/1) | 1} {periodic? (0/1) | 1} {width | 64} {height | 64} {render? (0/1) | 0}"
			<< std::endl;
		return -1;
	}

	tiles_dir = argv[1];
	if (argc > 2)
		tile_dim = atoi(argv[2]); // denotes tile dimension
	if (argc > 3)
		rotate = atoi(argv[3]); // 0 for no rotation, 1 for rotation
	if (argc > 4)
		periodic = atoi(argv[4]); // 0 if not periodic, 1 for periodic 
	if (argc > 5)
		width = atoi(argv[5]); // denotes tile width
	if (argc > 6)
		height = atoi(argv[6]); // denotes tile height;
	if (argc > 7) {
		out_name = argv[7];
	} else {
		out_name = "result.png";
	}
	if (argc > 8)
		render = atoi(argv[8]); // render toggle

	// The set of overlays describing how to compare two patterns. Stored
	// as an (x,y) shift. Shape: [O]
	std::vector<Pair> overlays;
	generate_neighbor_overlay(overlays);

	// The set of input images to use as templates. Shape: [T]
	std::vector<cv::Mat> template_imgs;
	load_tiles(tiles_dir, template_imgs);


	// The set of patterns/tiles taken from the input templates. Shape: [N]
	std::vector<cv::Mat> patterns;
	std::vector<int> counts;
	create_waveforms(template_imgs, tile_dim, rotate, patterns, counts);

	// Stores the set of allowed patterns for a given center pattern and
	// overlay. Stored like an adjacency list. Shape: [N, O][*]
	std::vector<std::vector<int>> fit_table;
	generate_fit_table(patterns, overlays, tile_dim, fit_table);

	Pair out_shape = Pair(width, height);
	CpuModel cpu_model(out_shape,
	            patterns.size(), overlays.size(),
	            tile_dim, periodic);
	GpuModel gpu_model(out_shape,
	            patterns.size(), overlays.size(),
	            tile_dim, periodic);

	// Shows all patterns
	if (render)
		show_patterns(cpu_model, patterns, overlays, fit_table);

	// Run and time CPU and GPU models
	auto t1 = std::chrono::high_resolution_clock::now();
	cpu_model.generate(overlays, counts, fit_table);
	auto t2 = std::chrono::high_resolution_clock::now();
	gpu_model.generate(overlays, counts, fit_table);
	auto t3 = std::chrono::high_resolution_clock::now();

	auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
	auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
	std::cout << "Total CPU time: " << cpu_time << std::endl;
	std::cout << "Total GPU time: " << gpu_time << std::endl;

	// Initialize blank output image
	cv::Mat result = cv::Mat(width, width, template_imgs[0].type());

	render_image(gpu_model, patterns, result);

	cv::resize(result, result, cv::Size(800, 800), 0.0, 0.0, cv::INTER_AREA);
	cv::imshow("result", result);
	cv::waitKey(0);

	std::ostringstream outputDir;
	outputDir << "results/" << out_name;
	cv::imwrite(outputDir.str(), result);
	std::cout << outputDir.str() << std::endl;

	return 0;
}
