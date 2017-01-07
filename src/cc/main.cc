// Copyright 2016, National University of Defense Technology
// Authors: Xuhao Chen <cxh@illinois.edu>
#include "common.h"
#include "graph_io.h"
#include "cc.h"
#include "verifier.h"

int main(int argc, char *argv[]) {
	printf("Connected Component with CUDA by Xuhao Chen\n");
	if (argc < 2) {
		printf("Usage: %s <graph> [device(0/1)]\n", argv[0]);
		exit(1);
	}
	int m, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
	W_TYPE *h_weight = NULL;
	read_graph(argc, argv, m, nnz, h_row_offsets, h_column_indices, h_degree, h_weight, true);
	CompT *h_comp = (CompT *)malloc(m * sizeof(CompT));

	CCSolver(m, nnz, h_row_offsets, h_column_indices, h_comp);
	CCVerifier(m, h_row_offsets, h_column_indices, h_comp);

	free(h_row_offsets);
	free(h_column_indices);
	free(h_degree);
	return 0;
}