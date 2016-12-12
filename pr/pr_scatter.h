#define BFS_VARIANT "scatter"
#include "cuda_launch_config.hpp"
#include "cutil_subset.h"
#define EPSILON 0.03
#define MAX_ITER 19
const float kDamp = 0.85;
#define BLKSIZE 128

__global__ void initialize(int m, float *cur_pagerank, float *next_pagerank) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < m) {
		cur_pagerank[id] = 1.0f / (float)m;
		next_pagerank[id] = (1.0f - kDamp) / (float)m;
	}
}

__global__ void scatter(int m, int *row_offsets, int *column_indices, float *cur_pagerank, float *next_pagerank) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			unsigned row_begin = row_offsets[src];
			unsigned row_end = row_offsets[src + 1];
			unsigned degree = row_end - row_begin;
			float value = kDamp * cur_pagerank[src] / (float)degree;
			for (unsigned offset = row_begin; offset < row_end; ++ offset) {
				int dst = column_indices[offset];
				atomicAdd(&next_pagerank[dst], value);
			}
		}
	}
}
/*
__global__ void reduce(int m, float *cur_pagerank, float *next_pagerank, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			float local_diff = abs(next_pagerank[src] - cur_pagerank[src]);
			atomicAdd(diff, local_diff);
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = (1.0f - kDamp) / (float)m;
		}
	}
}
//*/
///*
#include <cub/cub.cuh>
__global__ void reduce(int m, float *cur_pagerank, float *next_pagerank, float *diff) {
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	typedef cub::BlockReduce<float, BLKSIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	int total_inputs = (m - 1) / (gridDim.x * blockDim.x) + 1;
	float local_diff = 0;
	for (int src = tid; total_inputs > 0; src += blockDim.x * gridDim.x, total_inputs--) {
		if(src < m) {
			local_diff += abs(next_pagerank[src] - cur_pagerank[src]);
			cur_pagerank[src] = next_pagerank[src];
			next_pagerank[src] = (1.0f - kDamp) / (float)m;
		}
	}
	float block_sum = BlockReduce(temp_storage).Sum(local_diff);
	if(threadIdx.x == 0) atomicAdd(diff, block_sum);
}
//*/
void pr(int m, int nnz, int *d_row_offsets, int *d_column_indices, int *d_degree) {
	unsigned zero = 0;
	float *d_diff, h_diff, e = 0.1;
	float *d_cur_pagerank, *d_next_pagerank;
	double starttime, endtime, runtime;
	int iteration = 0;
	const int nthreads = BLKSIZE;
	int nblocks = (m - 1) / nthreads + 1;
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_cur_pagerank, m * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_next_pagerank, m * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_diff, sizeof(float)));
	initialize <<<nblocks, nthreads>>> (m, d_cur_pagerank, d_next_pagerank);
	CudaTest("initializing failed");

	//const size_t max_blocks_1 = maximum_residency(update_neighbors, nthreads, 0);
	const size_t max_blocks = maximum_residency(scatter, nthreads, 0);
	//const size_t max_blocks = 5;
	printf("Solving, max_blocks=%d, nblocks=%d, nthreads=%d\n", max_blocks, nblocks, nthreads);
	starttime = rtclock();
	do {
		++iteration;
		h_diff = 0.0f;
		CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(h_diff), cudaMemcpyHostToDevice));
		scatter <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, d_cur_pagerank, d_next_pagerank);
		CudaTest("solving kernel1 failed");
		reduce <<<nblocks, nthreads>>> (m, d_cur_pagerank, d_next_pagerank, d_diff);
		CudaTest("solving kernel2 failed");
		CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(h_diff), cudaMemcpyDeviceToHost));
		printf("iteration=%d, diff=%f\n", iteration, h_diff);
	} while (h_diff > EPSILON && iteration < MAX_ITER);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();
	printf("\titerations = %d.\n", iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", BFS_VARIANT, runtime);
	CUDA_SAFE_CALL(cudaFree(d_diff));
	return;
}