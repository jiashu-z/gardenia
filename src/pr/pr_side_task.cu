#include <argparse/argparse.hpp>
#include <iostream>
#include "task.h"
#include <thread>
#include <unistd.h>
#include <csignal>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <atomic>
#include "cutil_subset.h"
#include "worklistc.h"
#include <chrono>

#include "pr.h"
#include "timer.h"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"
#include <cub/cub.cuh>

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

// pull operation needs incoming neighbor list
__global__ void pull_fused(int m, const uint64_t *row_offsets,
                           const VertexId *column_indices,
                           ScoreT *scores, ScoreT *outgoing_contrib,
                           float *diff, ScoreT base_score) {
  typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
  int src = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float error = 0;
  if (src < m) {
    IndexT row_begin = row_offsets[src];
    IndexT row_end = row_offsets[src + 1];
    ScoreT incoming_total = 0;
    for (IndexT offset = row_begin; offset < row_end; ++offset) {
      IndexT dst = column_indices[offset];
      incoming_total += outgoing_contrib[dst];
    }
    ScoreT old_score = scores[src];
    scores[src] = base_score + kDamp * incoming_total;
    error += fabs(scores[src] - old_score);
  }
  float block_sum = BlockReduce(temp_storage).Sum(error);
  if (threadIdx.x == 0) atomicAdd(diff, block_sum);
}

void PRSolver(Graph &g, ScoreT *scores) {
  auto m = g.V();
  auto nnz = g.E();
  auto h_row_offsets = g.in_rowptr();
  auto h_column_indices = g.in_colidx();
  //print_device_info(0);
  uint64_t *d_row_offsets;
  VertexId *d_column_indices;
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_row_offsets, (m + 1) * sizeof(uint64_t)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_column_indices, nnz * sizeof(VertexId)));
  CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

  std::vector<VertexId> degrees(m);
  for (VertexId i = 0; i < m; i++) degrees[i] = g.get_degree(i);
  int *d_degrees;
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_degrees, m * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_degrees, &degrees[0], m * sizeof(int), cudaMemcpyHostToDevice));
  ScoreT *d_scores, *d_sums, *d_contrib;
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_scores, m * sizeof(ScoreT)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_sums, m * sizeof(ScoreT)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_contrib, m * sizeof(ScoreT)));
  CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
  float *d_diff, h_diff;
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_diff, sizeof(float)));

  int iter = 0;
  int nthreads = BLOCK_SIZE;
  int nblocks = (m - 1) / nthreads + 1;
  const ScoreT base_score = (1.0f - kDamp) / m;
  printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

  Timer t;
  t.Start();
  do {
    ++iter;
    h_diff = 0;
    CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
    contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
    CudaTest("solving kernel contrib failed");
#if FUSED
    pull_fused<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_scores, d_contrib, d_diff, base_score);
#else
    pull_step<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
    l1norm<<<nblocks, nthreads>>>(m, d_scores, d_sums, d_diff, base_score);
#endif
    CudaTest("solving kernel pull failed");
    CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
    printf(" %2d    %f\n", iter, h_diff);
  } while (h_diff > EPSILON && iter < MAX_ITER);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  printf("\titerations = %d.\n", iter);
  printf("\truntime [cuda_pull] = %f ms.\n", t.Millisecs());
  CUDA_SAFE_CALL(cudaMemcpy(scores, d_scores, m * sizeof(ScoreT), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_row_offsets));
  CUDA_SAFE_CALL(cudaFree(d_column_indices));
  CUDA_SAFE_CALL(cudaFree(d_degrees));
  CUDA_SAFE_CALL(cudaFree(d_scores));
  CUDA_SAFE_CALL(cudaFree(d_sums));
  CUDA_SAFE_CALL(cudaFree(d_contrib));
  CUDA_SAFE_CALL(cudaFree(d_diff));
  return;
}

class PrSideTask final : public BubbleBanditTask {
 private:

  bool with_profiler_;
  std::string file_type_;
  std::string graph_prefix_;
  std::string symmetrize_;
  std::atomic<bool> init_event_;
  std::atomic<bool> start_event_;
  std::atomic<bool> pause_event_;
  std::atomic<bool> stop_event_;
  std::atomic<bool> preempt_event_;
  std::atomic<int> counter_;
  std::atomic<int64_t> ts0_;
  std::atomic<int64_t> ts1_;
  std::thread runner_;
  double duration_;
  std::atomic<double> end_time_;

  auto do_i_have_enough_time() -> bool {
    auto current_time = get_current_time_in_micro();

    return end_time_ - current_time > duration_;
  }

 public:
  PrSideTask(int task_id,
             std::string name,
             std::string device,
             std::string scheduler_addr,
             bool with_profiler,
             std::string file_type,
             std::string graph_prefix,
             std::string symmetrize) : BubbleBanditTask(task_id, name, device, scheduler_addr) {
    with_profiler_ = with_profiler;
    file_type_ = file_type;
    graph_prefix_ = graph_prefix;
    symmetrize_ = symmetrize;
    init_event_ = false;
    start_event_ = false;
    pause_event_ = false;
    stop_event_ = false;
    preempt_event_ = false;
    counter_ = 0;
    ts0_ = 0;
    ts1_ = 0;
    duration_ = 0.1;
    end_time_ = 0.0;
  }

  int64_t init(int64_t task_id) override {
    assert(task_id == task_id_);
    std::cout << "Init task " << task_id << std::endl;
    init_event_ = true;
    return 0;
  }

  int64_t start(int64_t task_id, double end_time) override {
    assert(task_id == task_id_);
    std::cout << "Start task " << task_id << " with end time " << end_time << std::endl;
    end_time_ = end_time;
    start_event_ = true;
    return 0;
  }

  int64_t pause(int64_t task_id) override {
    assert(task_id == task_id_);
    std::cout << "Pause task " << task_id << std::endl;
    pause_event_ = true;
    return 0;
  }

  int64_t stop(int64_t task_id) override {
    assert(task_id == task_id_);
    std::cout << "Stop task " << task_id << std::endl;
    stop_event_ = true;
    runner_.join();
    std::cout << "Task " << task_id << " stopped" << std::endl;
    kill(getpid(), SIGINT);
    return 0;
  }

  int64_t preempt(int64_t task_id) override {
    assert(task_id == task_id_);
    std::cout << "Preempt task " << task_id << std::endl;
    preempt_event_ = true;
    return 0;
  }

  void run() override {
    auto device = device_.at(5) - '0';
    std::cout << "Device: " << device << std::endl;
    cudaSetDevice(device);
    if (with_profiler_) {
//      TODO
    }
    Graph g(graph_prefix_, file_type_, std::stoi(symmetrize_));
    auto m = g.V();
    auto nnz = g.E();
    auto h_row_offsets = g.in_rowptr();
    auto h_column_indices = g.in_colidx();
    uint64_t *d_row_offsets;
    VertexId *d_column_indices;
    int *d_degrees;
    ScoreT *d_scores, *d_sums, *d_contrib;
    float *d_diff, h_diff;

    std::vector<VertexId> degrees(m);
    for (VertexId i = 0; i < m; i++) {
      degrees[i] = g.get_degree(i);
    }
    int iter = 0;
    int nthreads = BLOCK_SIZE;
    int nblocks = (m - 1) / nthreads + 1;
    const ScoreT base_score = (1.0f - kDamp) / m;
    const ScoreT init_score = 1.0f / m;
    std::vector<ScoreT> scores_vec(m, init_score);
    auto scores = &scores_vec[0];
    printf("Launching CUDA PR solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);

    auto state = BubbleBanditTask::State::CREATED;

    while (true) {
      switch (state) {
        case BubbleBanditTask::State::CREATED: {
          if (init_event_) {
            init_event_ = false;
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_row_offsets, (m + 1) * sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_column_indices, nnz * sizeof(VertexId)));
            CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets,
                                      h_row_offsets,
                                      (m + 1) * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice))
            CUDA_SAFE_CALL(cudaMemcpy(d_column_indices,
                                      h_column_indices,
                                      nnz * sizeof(VertexId),
                                      cudaMemcpyHostToDevice))

            CUDA_SAFE_CALL(cudaMalloc((void **) &d_degrees, m * sizeof(int)));
            CUDA_SAFE_CALL(cudaMemcpy(d_degrees, &degrees[0], m * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_scores, m * sizeof(ScoreT)));
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_sums, m * sizeof(ScoreT)));
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_contrib, m * sizeof(ScoreT)));
            CUDA_SAFE_CALL(cudaMemcpy(d_scores, scores, m * sizeof(ScoreT), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMalloc((void **) &d_diff, sizeof(float)));
          }
          break;
        }
        case BubbleBanditTask::State::PENDING: {
          if (start_event_) {
            start_event_ = false;
            state = BubbleBanditTask::State::RUNNING;
            std::cout << "State from PENDING to RUNNING" << std::endl;
          } else if (preempt_event_) {
            preempt_event_ = false;
          }
          break;
        }
        case BubbleBanditTask::State::RUNNING: {
          if (pause_event_) {
            pause_event_ = false;
            state = BubbleBanditTask::State::PENDING;
            std::cout << "State from RUNNING to PENDING" << std::endl;
          } else if (preempt_event_) {
            preempt_event_ = false;
          } else {
            if (!do_i_have_enough_time()) {
              printf("I do not have enough time, current: %f, end: %f\n",
                     get_current_time_in_micro(),
                     end_time_.load());
              auto end_time = end_time_.load();
              if (end_time - get_current_time_in_micro() > 1000) {
                usleep((end_time - get_current_time_in_micro()) / 1000);
              }
            } else {
              ++iter;
              h_diff = 0;
              CUDA_SAFE_CALL(cudaMemcpy(d_diff, &h_diff, sizeof(float), cudaMemcpyHostToDevice));
              contrib <<<nblocks, nthreads>>>(m, d_scores, d_degrees, d_contrib);
              CudaTest("solving kernel contrib failed");
              pull_step<<<nblocks, nthreads>>>(m, d_row_offsets, d_column_indices, d_sums, d_contrib);
              l1norm<<<nblocks, nthreads>>>(m, d_scores, d_sums, d_diff, base_score);
              CudaTest("solving kernel pull failed");
              CUDA_SAFE_CALL(cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost));
              printf(" %2d    %f\n", iter, h_diff);
              if (!(h_diff > EPSILON && iter < MAX_ITER)) {
                goto BREAK_LOOP;
              }
            }
          }
          break;
        }
        default: {
          assert(false);
        }
      }
      if (stop_event_) {
        break;
      }
      usleep(10000);
    }
    BREAK_LOOP:
    if (with_profiler_) {
//    TODO
    }
    if (stop_event_) {
      stop_event_ = false;
    } else {
      scheduler_client_.finish_task(task_id_);
    }
  }

  void finish() override {

  }

  void start_runner() override {

  }
};

grpc::Server *server_ptr;

void signal_handler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";

  // cleanup and close up stuff here
  // terminate program
  server_ptr->Shutdown();

  exit(signum);
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("pr_side_task");
  program.add_argument("-n", "--name");
  program.add_argument("-s", "--scheduler_addr");
  program.add_argument("-i", "--task_id");
  program.add_argument("-d", "--device");
  program.add_argument("-a", "--addr");
  // TODO: Jiashu: Fix profiler flag
  // program.add_argument("-p", "--profiler");
  program.add_argument("-t", "--file_type");
  program.add_argument("-g", "--graph_prefix");
  program.add_argument("--symmetrize");
  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    return 1;
  }

  auto name = program.get<std::string>("--name");
  auto scheduler_addr = program.get<std::string>("--scheduler_addr");
  auto task_id = std::stoi(program.get<std::string>("--task_id"));
  auto device = program.get<std::string>("--device");
  auto addr = program.get<std::string>("--addr");
  // auto with_profiler = bool(std::stoi(program.get<std::string>("--profiler")));
  auto with_profiler = false;
  auto file_type = program.get<std::string>("--file_type");
  auto graph_prefix = program.get<std::string>("--graph_prefix");
  auto symmetrize = program.get<std::string>("--symmetrize");

  auto task = PrSideTask(task_id, name, device, scheduler_addr, with_profiler, file_type, graph_prefix, symmetrize);

  auto service = TaskServiceImpl(&task);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << addr << std::endl;
  task.start_runner();
  server->Wait();

  signal(SIGINT, signal_handler);

  return 0;
}