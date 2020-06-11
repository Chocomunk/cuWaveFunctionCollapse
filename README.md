# GPU Accelerated Wave Function Collapse: Procedural Tile Generation

This is a CUDA/C++ port of https://github.com/mxgmn/WaveFunctionCollapse.

All contained template images also come from https://github.com/mxgmn/WaveFunctionCollapse.

## Getting Started
To build the project with g++ and nvcc on linux, run:

`make build`

This will generate a `bin/wfc` executable. This repository comes with template images (credited above), which can be used to test the compiled executable. We have included 4 test cases, which can all be executed by running:

`make test`

The output images are stored in the `results/` folder. For each test case, it will generate two images: one for an example CPU run and one for a GPU run. 

#### Please Note:
The included test cases are kept small in size to ensure that they will run quickly enough. However, this does mean that it is more likely to generate infeasible results since the total number of neighbors is reduced. Please understand that generating results that are guaranteed to be feasible is NP-hard, so you may need to run it multiple times to get a "nice" result.

In addition, since the results are randomly generated, and are not guaranteed to be "nice" it is meaningless to compare GPU results with CPU results in order to check for correctness. We can, however, make statements about the difference between the two implementations, which is discussed in the results section below.

## Requirements
This project was most recently built with [OpenCV 4.3.0](https://docs.opencv.org/4.3.0/), which is the only dependency. On our systems, we installed OpenCV using the following command:

`sudo apt-get install libopencv-dev`

In the case that the compiler cannot find OpenCV, you may need to change the `$(OPENCV)` variable in the makefile to any of `opencv['',1,2,3,4]`.

## The WFC Algorithm

**Note:** This is just a brief description of the WFC algorithm. For a detailed explanation, see https://github.com/mxgmn/WaveFunctionCollapse, re-writing mxgmn's explanation while giving due credit would be redundant.

This is a procedural generation algorithm. The implementation contained in this repository operates on images; it generates an output that is locally similar to the input. Currently, the algorithm takes a single template image, cuts it up into multiple tiles, and gives it to the WFC algorithm as input. A simple modification for the future is to take existing tiles and neighboring constraints as direct inputs to the program.

The WFC model takes a set of states with neighboring constraints as input. The goal of the algorithm is to generate a board of tiles such that all tiles are collapsed to a single state, and the board satisfies all neighboring constraints.

All tiles in the output are initialized to a superposition of all states in the input. The algorithm procedurally collapses each tile to a single state via the following loop:
1. Perform a measurement (observation) on the tile of lowest entropy to collapse it to a single state.
2. Propagate the changes caused by the collapsed tile throughout the board. This step continues until all current constraints are satisfied, and there are no changes left to propagate.
3. Determine the tile of lowest entropy (least uncertainty on the states) for the next loop iteration.

## GPU Implementation
Our goal was to reimplement the Parallel WaveFunction Collapse algorithm in CUDA, utilizing a GPUs massively parallel architecture to improve the runtime of the algorithm. As we will discuss in our performance analysis, we were not able to achieve this goal in time for the project submission.

The CPU implementation of the WFC model has a decent runtime and high memory costs as the dimension of the 2D tile increases. Most of the runtime comes from the multiple steps over every state in every tile that the propagator may need to compute.

The advantage of working with a GPUs parallel architecture is that we can compute on many tiles at once. We implemented our own CUDA kernels for specific subroutines that we optimize through parallelism.

### Clear Kernel
To clear a board, all tiles must be set to perfect superpositions of all the states, with maximum entropy for each tile. Initializing an entire unobserved wave with a CPU serial implementation would require iteratively traversing over the entire board (and each state within each tile) to mark all states as possible. Using CUDA, we assign individual threads of the GPU to handle multiple tiles concurrently and update the states and entropy of all tiles at the same time.


### Lowest Entropy Kernel
On the CPU, we iterate over every tile to search for the tile of lowest entropy. This tile is selected to be collapsed into a single state through observation. The lowest entropy kernel accelerates this using a min-reduce approach that is able to extract a minimum index in addition to the value.

### Collapse Wave Kernel
When a wave is observed, it is then collapsed to a single tile; taking it completely out of superposition. On the CPU, we would need to iterate over every state in a given tile, and assign them as "allowed" or "disallowed" one-by-one. The collapse wave kernel optimizes this by updating all state in a tile concurrently, avoiding the loop over the tile.

### Update Waves Kernel
This is the most complex kernel, and also may lend itself to further optimizing given more time in the future. On a CPU, we use a Breadth-First approach to propagate a change throughout the board, which often needs to revisit the same tile to update it multiple times. In the update waves kernel, we assign individual threads to a tile, and update their states concurrently. Specifically, we check each state within that tile to see if it has at least 1 compatible neighbor on all sides. If so, it remains allowed. Otherwise, we mark the state as disallowed.

We follow by applying a reduction over all tiles to determine if any tile in the board has changed. We continue updating the board through repeated calls of this kernel until no more changes occur, meaning that our state has completely settled.

### Compute Entropies Kernel
After updating the states of all the tiles, we then need to recompute their entropies. We avoid assigning the entropy during the propagation since we want to reduce global memory operations; it was already difficult enough to get the update waves kernel to it's current state on time. This kernel simply counts the number of allowed states per each tile, with each tile being operated on concurrently.

### IsCollapsed Kernel
The "stopping-condition" for this algorithm is when all tiles have either collapsed to a single state or cannot satisfy their constraints (they have no allowed tiles and are infeasible). On the CPU, we would simply need to iterate over all the tiles and check that no tile is eligible for collapse. On the GPU, we can use a reduction over all the tiles to determine whether any tile is still in a superposition. If no tiles are left is superposition, then our board has fully collapsed.

## Code Structure
The cuWFC program is implemented as if it were a library that you could install or download to use in your own projects to procedurally generate some data.

### Core Components
The Core of this algorithm consists of only the models and utilities for the algorithm. All models are defined in `src/models.h` and implemented primarily in `src/models.cpp`. For the GPU implementation, wrapper functions for our CUDA kernels are defined in `src/cuwfc.cuh`, and they are implemented alongside the kernels that the wrap in `src/cuwfc.cu`. All utility functions and types are defined in `src/wfc_util.h`, and are implemented in `src/wfc_util.cpp`.

### I/O
We viewed IO as a non-integral and modular part of our program. Ideally, a developer can write their own operations to convert an input into a set of states and neighboring rules. In addition, we allow them to generate their own outputs by providing access to superpositions of specific tiles. In this repository we provide a simple input and output for loading images for the sake of testing and demonstration. Both input and output are handled using OpenCV to convert images to and from CV matrices. All example input functions are defined in `src/input.h`, and implemented in `src/input.cpp`. Similarly, all example output functions are defined in `src/output.h`, and implemented in `src/output.cpp`.

### Models
To add support for extension to different types of models beyond the CPU and GPU implementations we already provide, we define a base `Model` class which acts as an abstract super-class for all model definitions. Consequently, `CpuModel` and `GpuModel` are subclasses which extend from `Model`. All `Model` instances are required to support the following functions: `generate(overlays, fit_table)`, `get_superposition(row, col, out_idxs)`, and `clear(fit_table)`.

### Integration
All interfaces for creating and running a model, as well as it's utilities, can me accessed by including the `src/wfc.h` file. This is  simply a super-header that includes the necessary sub-headers that allow full usage of this WaveFunctionCollapse algorithm. The given tests in `src/test.cpp` are a great example of simple instantiation and generation.

## Results
In this section, we present some images generated by our CPU and GPU implementations. The first set of 8 images are all 16x16 images generated from small inputs.

###  Generated GPU Images
Below are standard results for the GPU implementation:

#### Red
![red_gpu][red_gpu]
#### Spirals
![spi_gpu][spi_gpu]
#### Dungeons
![dun_gpu][dun_gpu]
#### Paths
![pat_gpu][pat_gpu]

### Generate CPU Images
In comparson, here are standard results for the CPU implementation for the same patterns:

#### Red
![red_cpu][red_cpu]
#### Spirals
![spi_cpu][spi_cpu]
#### Dungeons
![dun_cpu][dun_cpu]
#### Paths
![pat_cpu][pat_cpu]

### Discussion
Its very interesting to note that the GPU returns a "nice" result more often than the cpu on this low-dimensional output. This is likely due to the difference in their implementations: the GPU algorithm operates on all tiles at the same time while the CPU "fans out". By operating on all tiles concurrently, and only updating the wave states after checking all neighbors, the GPU implementation is able to ensure that the whole board sees a reasonable propagation at each step. In comparison, the CPU fully updates a wave whenever it encounters it during the Breadth-First exploration, which doees not take the entire board's state into consideration. Given the reduced number of neighbors, it becomes harder for the CPU implementation to satisfy the constraints, as any strong change to one tile seriously affects its neighbors as well.

In comparison, we have also generated CPU images on much larger 64 x 64 outputs. On these larger outputs, the immediate propagation in the CPU doees not have as strong of an effect throughout the entire board, allowing it so successfully collapse to a board state that (mostly) satisfies the neighboring constraints.

#### Red
![red_cpu1][red_cpu1]
#### Spirals
![spi_cpu1][spi_cpu1]
#### Dungeons
![dun_cpu1][dun_cpu1]
#### Paths
![pat_cpu1][pat_cpu1]

## Performance
Running the provided 4 test cases on a GTX 1070 yielded the following times:

| Algorithm     | Red   | Spirals | Dungeons | Paths  |
|---------------|-------|---------|----------|--------|
| CPU Time (ms) | 2174  | 10407   | 8539     | 7808   |
| GPU Time (ms) | 56560 | 582111  | 884087   | 784464 |
|               |       |         |          |        |

As can be seen from the table above, our GPU implementation failed horribly in all test cases.

### Analysis and Possible Improvements
To our disappointment, we were not able to fully optimize the GPU implementation in time for the submission date. Specifically, the GPU implementation runs much slower than the CPU implementation. A major reason for this is the inefficiency of the Update Waves kernel. A tile must check all 4 neighbors, meaning that we cannot access all neighbors in the same cache without a major increase to the memory usage. In addition, the kernel scales very poorly with the number of states. Because it must check every pair of states for each neighbor, inputs with a large number of states will quickly scale in runtime; especially given a GPU's poor sequential computational performance. In addition, we must make frequent accesses to other arrays in global memory to determine the position of the neighbor and to determine whether two states fit. It is infeasible to place much of this data into shared memory due to the problem of scaling number of states explained above. We do so anyway to squeeze some performance, but we then had to limit our test cases to ones with a small number of states within the input.

We have considered the following improvements to performance, that we would have attempted to implement given more time to work on this project. Since the states are stored as boolean values within each tile (either allowed or not allowed), we can compact tiles by storing them as bit-strings in an integral value. This should improve the cache-hitting of the kernel with the massively reduced space usage; however only up to a certain degree, as it is still possible for a large number of states to scale our kernel out of proportion. In addition, we were hoping to engineer a more intelligent representation for the fit tables that would improve the runtime to determine if two states fit properly with each-other. There may exist a way to only compare a list with its compatible neighbors while not introducing too much branching and thread divergence. Finally, one that we could have done but were hesitant to, is manually unrolling and implementing each neighbor comparison  to reduce reads to global memory. The reason we decided not to implement this was because we intended for our algorithm to be able to modularly support many different overlays between tiles. It might be the case that a developer needs two tiles far apart in the board to be correlated regarding their states.

We also list a few other slight improvements that we could still make given a short amount of extra time. Our reductions are not as perfectly optimized as they could be. For reductions that are fully optimized on their blocksize (as described by Mark Harris), we atomically accumulate the final values over all blocks. Instead, we could loop the reduction until we accumulate the final value in the last block. For reductions that do repeat until the last block (which we implemented out of necessity, as we could not atomically extract a minimum index), we may still optimize their implementations on their blocksize. In addition, we can try to minimize extraneous small memory transfers throughout the algorithm (mainly GPU and CPU communication variables) by using the internal workspace array in-place of many device-allocated variables.

## Sample Output
Below is the console output of an example run of the provided tests on a GTX 1070:

```
Running Red Tests:
bin/wfc tiles/red/ 2 1 1 16 16 0 0 red
Patterns: 12
Overlay Count: 4
Wave Shape: 15 x 15
Patterns: 12
Overlay Count: 4
Wave Shape: 15 x 15
Called Generate
Finished Algorithm in 224 iterations
Called Generate
Finished Algorithm in 65 iterations
Total CPU time: 2174
Total GPU time: 56560
Finished Rendering
results/red_cpu.png
Finished Rendering
results/red_gpu.png

Running Spirals Tests:
bin/wfc tiles/spirals/ 3 1 1 16 16 0 0 spirals
Patterns: 69
Overlay Count: 4
Wave Shape: 14 x 14
Patterns: 69
Overlay Count: 4
Wave Shape: 14 x 14
Called Generate
Finished Algorithm in 191 iterations
Called Generate
Finished Algorithm in 20 iterations
Total CPU time: 10407
Total GPU time: 582111
Finished Rendering
results/spirals_cpu.png
Finished Rendering
results/spirals_gpu.png

Running Dungeons Tests:
bin/wfc tiles/dungeons/ 3 0 1 16 16 0 0 dungeons
Patterns: 57
Overlay Count: 4
Wave Shape: 14 x 14
Patterns: 57
Overlay Count: 4
Wave Shape: 14 x 14
Called Generate
Finished Algorithm in 195 iterations
Called Generate
Finished Algorithm in 59 iterations
Total CPU time: 8539
Total GPU time: 884087
Finished Rendering
results/dungeons_cpu.png
Finished Rendering
results/dungeons_gpu.png

Running Paths Tests:
bin/wfc tiles/paths/ 3 0 1 16 16 0 0 paths
Patterns: 49
Overlay Count: 4
Wave Shape: 14 x 14
Patterns: 49
Overlay Count: 4
Wave Shape: 14 x 14
Called Generate
Finished Algorithm in 186 iterations
Called Generate
Finished Algorithm in 75 iterations
Total CPU time: 7808
Total GPU time: 784464
Finished Rendering
results/paths_cpu.png
Finished Rendering
results/paths_gpu.png
```

[dun_cpu]: images/dungeon_cpu.png
[pat_cpu]: images/paths_cpu.png
[red_cpu]: images/red_cpu.png
[spi_cpu]: images/spirals_cpu.png
[dun_gpu]: images/dungeon_gpu.png
[pat_gpu]: images/paths_gpu.png
[red_gpu]: images/red_gpu.png
[spi_gpu]: images/spirals_gpu.png
[dun_cpu1]: images/dungeon_cpu1.png
[pat_cpu1]: images/paths_cpu1.png
[red_cpu1]: images/red_cpu1.png
[spi_cpu1]: images/spirals_cpu1.png