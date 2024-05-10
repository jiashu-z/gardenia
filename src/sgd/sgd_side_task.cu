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
#include <cub/cub.cuh>
#include "graph_io.h"
#include "common.h"
#include <random>
#include "cutil_subset.h"

#define K (20)

#define SGD_VARIANT "base"
void read_graph_from_file_name(::std::string file_name,
                               int &m,
                               int &n,
                               int &nnz,
                               IndexT *&row_offsets,
                               IndexT *&column_indices,
                               int *&degree,
                               WeightT *&weight,
                               bool is_symmetrize = false,
                               bool is_transpose = false,
                               bool sorted = true,
                               bool remove_selfloops = true,
                               bool remove_redundents = true) {
  //if(is_symmetrize) printf("Requiring symmetric graphs for this algorithm\n");
  auto file_name_c_str = file_name.c_str();
  auto buf = new char[file_name.size() + 1]{'\0'};
  memmove(buf, file_name_c_str, file_name.size());
  if (strstr(buf, ".mtx")) {
    mtx2csr(buf,
            m,
            n,
            nnz,
            row_offsets,
            column_indices,
            weight,
            is_symmetrize,
            is_transpose,
            sorted,
            remove_selfloops,
            remove_redundents);
  } else if (strstr(buf, ".graph")) {
    graph2csr(buf,
              m,
              nnz,
              row_offsets,
              column_indices,
              weight,
              is_symmetrize,
              is_transpose,
              sorted,
              remove_selfloops,
              remove_redundents);
  } else if (strstr(buf, ".gr")) {
    gr2csr(buf,
           m,
           nnz,
           row_offsets,
           column_indices,
           weight,
           is_symmetrize,
           is_transpose,
           sorted,
           remove_selfloops,
           remove_redundents);
  } else {
    printf("Unrecognizable input file format\n");
    exit(0);
  }

  printf("Calculating degree...");
  degree = (int *) malloc(m * sizeof(int));
  for (int i = 0; i < m; i++) {
    degree[i] = row_offsets[i + 1] - row_offsets[i];
  }
  printf(" Done\n");
}

void Initialize(int len, LatentT *lv) {
  std::default_random_engine rng;
  std::uniform_real_distribution<float> dist(0, 0.1);
  for (int i = 0; i < len; ++i) {
    for (int j = 0; j < K; ++j) {
      lv[i * K + j] = dist(rng);
    }
  }
}

typedef cub::BlockReduce<ScoreT, BLOCK_SIZE> BlockReduce;
__global__ void update(int m,
                       int n,
                       int *row_offsets,
                       int *column_indices,
                       ScoreT *rating,
                       LatentT *user_lv,
                       LatentT *item_lv,
                       ScoreT lambda,
                       ScoreT step,
                       int *ordering,
                       ScoreT *squared_errors) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m) {
    //int user_id = ordering[tid];
    int user_id = tid;
    int row_begin = row_offsets[user_id];
    int row_end = row_offsets[user_id + 1];
    int user_offset = K * user_id;
    LatentT *ulv = &user_lv[user_offset];
    for (int offset = row_begin; offset < row_end; ++offset) {
      int item_id = column_indices[offset];
      int item_offset = K * item_id;
      LatentT *ilv = &item_lv[item_offset];
      ScoreT estimate = 0;
      for (int i = 0; i < K; i++)
        estimate += ulv[i] * ilv[i];
      ScoreT delta = rating[offset] - estimate;
      squared_errors[user_id] += delta * delta;
      for (int i = 0; i < K; i++) {
        LatentT p_u = ulv[i];
        LatentT p_i = ilv[i];
        ulv[i] += step * (-lambda * p_u + p_i * delta);
        ilv[i] += step * (-lambda * p_i + p_u * delta);
      }
    }
  }
}

__global__ void rmse(int m, ScoreT *squared_errors, ScoreT *total_error) {
  int uid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  ScoreT local_error = 0.0;
  if (uid < m) local_error = squared_errors[uid];
  ScoreT block_sum = BlockReduce(temp_storage).Sum(local_error);
  if (threadIdx.x == 0) atomicAdd(total_error, block_sum);
}

class SgdSideTask final : public BubbleBanditTask {
 private:
  std::string file_name_;
  double lambda_;
  double step_;
  int max_iter_;
  double epsilon_;

  int m, n, nnz, *h_row_offsets = nullptr, *h_column_indices = nullptr, *h_degree = nullptr;
  WeightT *h_weight = nullptr;

  int *d_row_offsets, *d_column_indices;

  LatentT *h_user_lv;
  LatentT *h_item_lv;
  LatentT *lv_u;
  LatentT *lv_i;
  ScoreT *h_rating;
  int *ordering;
  std::atomic<int> iter;

  int nthreads;
  int nblocks;
  ScoreT *d_rating;
  int *d_ordering;
  LatentT *d_user_lv, *d_item_lv;
  ScoreT h_error, *d_error, *squared_errors;
 public:
  SgdSideTask(int task_id,
              std::string name,
              std::string device,
              std::string scheduler_addr,
              int profiler_level,
              std::string file_name,
              double lambda,
              double step,
              int max_iter,
              double epsilon,
              double duration)
      : BubbleBanditTask(task_id, name, device, scheduler_addr, profiler_level) {
    file_name_ = file_name;
    lambda_ = lambda;
    step_ = step;
    max_iter_ = max_iter;
    epsilon_ = epsilon;
    duration_ = duration;
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

  auto submitted_to_created() -> void override {
    auto device = device_.at(5) - '0';
    std::cout << "Device: " << device << std::endl;
    cudaSetDevice(device);

    read_graph_from_file_name(file_name_,
                              m,
                              n,
                              nnz,
                              h_row_offsets,
                              h_column_indices,
                              h_degree,
                              h_weight,
                              false,
                              false,
                              false,
                              false,
                              false);

    printf("num_users=%d, num_items=%d\n", m, n);
    printf("regularization_factor=%f, learning_rate=%f\n", lambda_, step_);
    printf("max_iter=%d, epsilon=%f\n", max_iter_, epsilon_);

    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    h_user_lv = (LatentT *) malloc(m * K * sizeof(LatentT));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    h_item_lv = (LatentT *) malloc(n * K * sizeof(LatentT));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    lv_u = (LatentT *) malloc(m * K * sizeof(LatentT));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    lv_i = (LatentT *) malloc(n * K * sizeof(LatentT));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;

    Initialize(m, lv_u);
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    Initialize(n, lv_i);
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    for (int i = 0; i < m * K; i++) {
      h_user_lv[i] = lv_u[i];
    }
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    for (int i = 0; i < n * K; i++) {
      h_item_lv[i] = lv_i[i];
    }
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    for (int i = 0; i < nnz; i++) {
      h_rating[i] = (ScoreT) h_weight[i];
    }
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    ordering = nullptr;
    std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  }

  auto created_to_paused() -> void override {

    auto &num_users = m;
    auto &num_items = n;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_row_offsets, (num_users + 1) * sizeof(int)))
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_column_indices, nnz * sizeof(int)))
    CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (num_users + 1) * sizeof(int), cudaMemcpyHostToDevice))
    CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(int), cudaMemcpyHostToDevice))
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_rating, nnz * sizeof(ScoreT)))
    CUDA_SAFE_CALL(cudaMemcpy(d_rating, h_rating, nnz * sizeof(ScoreT), cudaMemcpyHostToDevice))
    //CUDA_SAFE_CALL(cudaMalloc((void **)&d_ordering, num_users * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMemcpy(d_ordering, h_ordering, num_users * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc((void **) &d_user_lv, num_users * K * sizeof(LatentT)))
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_item_lv, num_items * K * sizeof(LatentT)))
    CUDA_SAFE_CALL(cudaMemcpy(d_user_lv, h_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyHostToDevice))
    CUDA_SAFE_CALL(cudaMemcpy(d_item_lv, h_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyHostToDevice))
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_error, sizeof(ScoreT)))
    CUDA_SAFE_CALL(cudaMalloc((void **) &squared_errors, num_users * sizeof(ScoreT)))
    CUDA_SAFE_CALL(cudaMemset(d_error, 0, sizeof(ScoreT)))

    iter = 0;
    nthreads = BLOCK_SIZE;
    nblocks = (num_users - 1) / nthreads + 1;
    printf("Launching CUDA SGD solver (%d CTAs, %d threads/CTA) ...\n", nblocks, nthreads);
    CUDA_SAFE_CALL(cudaDeviceSynchronize())

  }

  auto paused_to_running() -> void override {
  }

  auto running_to_paused() -> void override {
  }

  auto running_to_finished() -> void override {
  }

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
    // return iter >= max_iter_ || h_error <= epsilon_;
  }

  auto step() -> void override {
    ++iter;
    h_error = 0.0;
    auto &num_users = m;
    auto &num_items = n;
    CUDA_SAFE_CALL(cudaMemset(squared_errors, 0, num_users * sizeof(ScoreT)));
    CUDA_SAFE_CALL(cudaMemcpy(d_error, &h_error, sizeof(ScoreT), cudaMemcpyHostToDevice));
    update<<<nblocks, nthreads>>>(num_users,
                                  num_items,
                                  d_row_offsets,
                                  d_column_indices,
                                  d_rating,
                                  d_user_lv,
                                  d_item_lv,
                                  lambda_,
                                  step_,
                                  d_ordering,
                                  squared_errors);
    // CudaTest("solving kernel update failed");
    rmse<<<nblocks, nthreads>>>(num_users, squared_errors, d_error);
    // CudaTest("solving kernel rmse failed");
    CUDA_SAFE_CALL(cudaMemcpy(&h_error, d_error, sizeof(ScoreT), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    //printf("h_error=%f\n", h_error);
    printf("iteration %d: RMSE error = %f\n", iter.load(), sqrt(h_error / nnz));
    //CUDA_SAFE_CALL(cudaMemcpy(h_user_lv, d_user_lv, num_users * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
    //CUDA_SAFE_CALL(cudaMemcpy(h_item_lv, d_item_lv, num_items * K * sizeof(LatentT), cudaMemcpyDeviceToHost));
    //print_latent_vector(num_users, num_items, h_user_lv, h_item_lv);
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
  argparse::ArgumentParser program("sgd_side_task");
  program.add_argument("-n", "--name");
  program.add_argument("-s", "--scheduler_addr");
  program.add_argument("-i", "--task_id");
  program.add_argument("-d", "--device");
  program.add_argument("-a", "--addr");
  program.add_argument("--duration");
  program.add_argument("--profiler_level");
  program.add_argument("--graph_file");
  program.add_argument("--lambda");
  program.add_argument("--step");
  program.add_argument("--max_iter");
  program.add_argument("--epsilon");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

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
  auto profiler_level = std::stoi(program.get<std::string>("--profiler_level"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto file_name = program.get<std::string>("--graph_file");
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto lambda = std::stod(program.get<std::string>("--lambda"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto step = std::stod(program.get<std::string>("--step"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto max_iter = std::stoi(program.get<std::string>("--max_iter"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;
  auto epsilon = std::stod(program.get<std::string>("--epsilon"));
  std::cout << __FILE__ << ":" << __LINE__ << std::endl;

  auto task = SgdSideTask(task_id, name, device, scheduler_addr, profiler_level,
                          file_name, lambda, step, max_iter, epsilon, duration);
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
