#include <argparse/argparse.hpp>
#include <iostream>
#include "task.h"
#include <thread>
#include <csignal>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <atomic>
#include "cutil_subset.h"

#include "pr.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>
#include <fstream>
#include <cassert>

#define FUSED 0
#define PR_VARIANT "pull"

typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;

__global__ void contrib(int m, ScoreT *scores, int *degree, ScoreT *outgoing_contrib) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < m) outgoing_contrib[u] = scores[u] / degree[u];
}

__global__ void pull_step(int m, const uint64_t *row_offsets,
                          const VertexId *column_indices,
                          ScoreT *sums, ScoreT *outgoing_contrib) {
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst < m) {
    IndexT row_begin = row_offsets[dst];
    IndexT row_end = row_offsets[dst + 1];
    ScoreT incoming_total = 0;
    for (IndexT offset = row_begin; offset < row_end; ++offset) {
      //IndexT src = column_indices[offset];
      IndexT src = __ldg(column_indices + offset);
      //incoming_total += outgoing_contrib[src];
      incoming_total += __ldg(outgoing_contrib + src);
    }
    sums[dst] = incoming_total;
  }
}

__global__ void l1norm(int m, ScoreT *scores, ScoreT *sums,
                       float *diff, ScoreT base_score) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float local_diff = 0;
  if (u < m) {
    ScoreT new_score = base_score + kDamp * sums[u];
    local_diff += fabs(new_score - scores[u]);
    scores[u] = new_score;
    sums[u] = 0;
  }
  float block_sum = BlockReduce(temp_storage).Sum(local_diff);
  if (threadIdx.x == 0) atomicAdd(diff, block_sum);
}

class PrSideTask final : public BubbleBanditTask {
 private:

  std::string file_type_;
  std::string graph_prefix_;
  std::string symmetrize_;
  int max_iter_;

  Graph *g_ptr;
  VertexId m;
  size_t nnz;
  uint64_t *h_row_offsets;
  VertexId *h_column_indices;
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  int *d_degrees;
  ScoreT *d_scores, *d_sums, *d_contrib;
  float *d_diff, h_diff;

  std::vector<VertexId> degrees;
  std::atomic<int> iter;
  int nthreads;
  int nblocks;
  ScoreT base_score;
  ScoreT init_score;
  std::vector<ScoreT> scores_vec;
  float *scores;

 public:
  PrSideTask(int task_id,
             std::string name,
             std::string device,
             std::string scheduler_addr,
             double duration,
             int profiler_level,
             std::string file_type,
             std::string graph_prefix,
             std::string symmetrize,
             int max_iter) : BubbleBanditTask(task_id, name, device, scheduler_addr, profiler_level) {
    file_type_ = file_type;
    graph_prefix_ = graph_prefix;
    symmetrize_ = symmetrize;
    max_iter_ = max_iter;
    duration_ = duration;
  }

  auto submitted_to_created() -> void override {
    auto device = device_.at(5) - '0';
    printf("device_: %s, CUDA:%d\n", device_.c_str(), device);
    cudaSetDevice(device);
    g_ptr = new Graph(graph_prefix_, file_type_, std::stoi(symmetrize_), 1);
    auto &g = *g_ptr;
    m = g.V();
    nnz = g.E();
    h_row_offsets = g.in_rowptr();
    h_column_indices = g.in_colidx();

    degrees = std::vector<VertexId>(m);
    for (VertexId i = 0; i < m; i++) {
      degrees[i] = g.get_degree(i);
    }
    iter = 0;
    nthreads = BLOCK_SIZE;
    nblocks = (m - 1) / nthreads + 1;
    base_score = (1.0f - kDamp) / m;
    init_score = 1.0f / m;
    scores_vec = std::vector<ScoreT>(m, init_score);
    scores = &scores_vec[0];
    printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

    state_ = BubbleBanditTask::State::CREATED;
  }

  auto created_to_paused() -> void override {
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_row_offsets, (m + 1) * sizeof(uint64_t)));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_column_indices, nnz * sizeof(VertexId)));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    printf("Size of d_row_offsets: %ld\n", (m + 1) * sizeof(uint64_t));
    CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets,
                              h_row_offsets,
                              (m + 1) * sizeof(uint64_t),
                              cudaMemcpyHostToDevice))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(d_column_indices,
                              h_column_indices,
                              nnz * sizeof(VertexId),
                              cudaMemcpyHostToDevice))

    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_degrees, m * sizeof(int)))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(d_degrees, &degrees[0], m * sizeof(int), cudaMemcpyHostToDevice))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_scores, m * sizeof(ScoreT)))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_sums, m * sizeof(ScoreT)))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_contrib, m * sizeof(ScoreT)))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_diff, sizeof(float)))
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  }

  auto paused_to_running() -> void override {
  }

  auto running_to_paused() -> void override {
  }

  auto running_to_finished() -> void override {
  }

/*

    def to_stopped(self) -> None:
        with open(f"./{self.task_name}_{self.task_id}_side_task.txt", "w") as f:
            f.write(str(self.step_counter * self.batch_size))
            f.flush()
*/

  auto to_stopped() -> void override {
    std::string out_file_name = name_ + "_" + std::to_string(task_id_) + "_side_task.txt";
    std::ofstream out_file(out_file_name);
    assert(out_file.is_open());
    out_file << iter;
    out_file.flush();
    out_file.close();
  }

  auto is_finished() -> bool override {
    return max_iter_ != 0 && iter >= max_iter_;
  }

  auto step() -> void override {
    ++iter;
    h_diff = 0;
    CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
    contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
    // CudaTest("solving kernel contrib failed");
    pull_step<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
    l1norm<<<nblocks, nthreads>>>(m, d_scores, d_sums, d_diff, base_score);
    // CudaTest("solving kernel pull failed");
    CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    printf(" %2d    %f\n", iter.load(), h_diff);
  }

};

grpc::Server *server_ptr;

void signal_handler(int signum) {
  printf("Received signal %d\n", signum);

  // cleanup and close up stuff here
  // terminate program
  sleep(1);
  server_ptr->Shutdown();
  printf("Exit task\n");

  exit(0);
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("pr_side_task");
  program.add_argument("-n", "--name");
  program.add_argument("-s", "--scheduler_addr");
  program.add_argument("-i", "--task_id");
  program.add_argument("-d", "--device");
  program.add_argument("-a", "--addr");
  program.add_argument("--duration");
  program.add_argument("--profiler_level");
  program.add_argument("-t", "--file_type");
  program.add_argument("-g", "--graph_prefix");
  program.add_argument("--symmetrize");
  program.add_argument("--max_iter");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  auto name = program.get<std::string>("--name");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto scheduler_addr = program.get<std::string>("--scheduler_addr");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto task_id = std::stoi(program.get<std::string>("--task_id"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto device = program.get<std::string>("--device");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto addr = program.get<std::string>("--addr");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto duration = std::stod(program.get<std::string>("--duration"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto profiler_level = std::stoi(program.get<std::string>("--profiler_level"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto file_type = program.get<std::string>("--file_type");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto graph_prefix = program.get<std::string>("--graph_prefix");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto symmetrize = program.get<std::string>("--symmetrize");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto max_iter = std::stoi(program.get<std::string>("--max_iter"));

  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto task =
      PrSideTask(task_id, name, device, scheduler_addr, duration, profiler_level, file_type, graph_prefix, symmetrize, max_iter);
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  signal(SIGINT, signal_handler);

  auto service = TaskServiceImpl(&task);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << addr << std::endl;
  task.start_runner();
  server->Wait();

  return 0;
}