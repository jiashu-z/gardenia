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
#include <iostream>

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
  bool with_profiler_;
  ::std::string file_name_;
  double lambda_;
  double step_;
  int max_iters_;
  double epsilon_;
  ::std::atomic<bool> init_event_;
  ::std::atomic<bool> start_event_;
  ::std::atomic<bool> pause_event_;
  ::std::atomic<bool> stop_event_;
  ::std::atomic<bool> preempt_event_;
  ::std::atomic<int> counter_;
  ::std::atomic<int64_t> ts0_;
  ::std::atomic<int64_t> ts1_;
  ::std::thread runner_;
  double duration_;
  ::std::atomic<double> end_time_;

  auto do_i_have_enough_time() -> bool {
    // Get the current time in microseconds
    auto current_time = get_current_time_in_micro();

    // Check if the current time is less than the end time
    return end_time_ - current_time > duration_;

  }

 public:
  SgdSideTask(int task_id,
              std::string name,
              std::string device,
              std::string scheduler_addr,
              bool with_profiler,
              std::string file_name,
              std::string lambda,
              std::string step,
              std::string max_iters,
              std::string epsilon)
      : BubbleBanditTask(task_id, name, device, scheduler_addr) {
    with_profiler_ = with_profiler;
    file_name_ = file_name;
    lambda_ = std::stof(lambda);
    step_ = std::stof(step);
    max_iters_ = std::stod(max_iters);
    epsilon_ = std::stof(epsilon);
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
      // TODO: You probably want to add some profiling logic here.
    }

    int m, n, nnz, *h_row_offsets = NULL, *h_column_indices = NULL, *h_degree = NULL;
    WeightT *h_weight = NULL;
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
    printf("max_iters=%d, epsilon=%f\n", max_iters_, epsilon_);

    LatentT *h_user_lv = (LatentT *) malloc(m * K * sizeof(LatentT));
    LatentT *h_item_lv = (LatentT *) malloc(n * K * sizeof(LatentT));
    LatentT *lv_u = (LatentT *) malloc(m * K * sizeof(LatentT));
    LatentT *lv_i = (LatentT *) malloc(n * K * sizeof(LatentT));
    ScoreT *h_rating = (ScoreT *) malloc(nnz * sizeof(ScoreT));

    Initialize(m, lv_u);
    Initialize(n, lv_i);
    for (int i = 0; i < m * K; i++) h_user_lv[i] = lv_u[i];
    for (int i = 0; i < n * K; i++) h_item_lv[i] = lv_i[i];
    for (int i = 0; i < nnz; i++) h_rating[i] = (ScoreT) h_weight[i];
    int *ordering = NULL;

    auto state = BubbleBanditTask::State::CREATED;
    while (true) {
      switch (state) {
        case BubbleBanditTask::State::CREATED: {
          if (init_event_) {
            init_event_ = false;

            state = BubbleBanditTask::State::PENDING;
            std::cout << "State from CREATED to PENDING" << std::endl;
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
            // TODO: Fully clear GPU memory.

            state = BubbleBanditTask::State::CREATED;
            std::cout << "State from PENDING to CREATED" << std::endl;
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

            state = BubbleBanditTask::State::CREATED;
            std::cout << "State from RUNNING to CREATED" << std::endl;
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
              if (true) {
                // TODO: clean up.
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
      // TODO: You probably want to add some profiling logic here.
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
    std::cout << "Start runner of task " << task_id_ << std::endl;
    runner_ = std::thread([this] { run(); });
  }
};

grpc::Server *server_ptr;

void signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";

  // cleanup and close up stuff here
  // terminate program
  server_ptr->Shutdown();

  exit(signum);
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("program_name");

  auto name = program.get<std::string>("--name");
  auto scheduler_addr = program.get<std::string>("--scheduler_addr");
  auto task_id = std::stoi(program.get<std::string>("--task_id"));
  auto device = program.get<std::string>("--device");
  auto addr = program.get<std::string>("--addr");
  // auto with_profiler = bool(std::stoi(program.get<std::string>("--profiler")));
  auto with_profiler = false;
  auto file_name = program.get<std::string>("--file_name");
  auto lambda = program.get<std::string>("--lambda");
  auto step = program.get<std::string>("--step");
  auto max_iters = program.get<std::string>("--max_iters");
  auto epsilon = program.get<std::string>("--epsilon");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto task = SgdSideTask(task_id, name, device, scheduler_addr, with_profiler,
                          file_name, lambda, step, max_iters, epsilon);

  // task.init(task_id);
  // task.run();
  // task.stop(task_id);
  auto service = TaskServiceImpl(&task);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << addr << std::endl;
  task.start_runner();
  server->Wait();

  signal(SIGINT, signalHandler);

  return 0;
}
