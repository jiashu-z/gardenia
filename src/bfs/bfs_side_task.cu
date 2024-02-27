#include "bfs.h"
#include <argparse/argparse.hpp>
#include <iostream>
#include "task.h"
#include <thread>
#include <unistd.h>
#include <csignal>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <atomic>
#include "cutil_subset.h"
#include "worklistc.h"
#include <chrono>

__global__ void insert(int source, Worklist2 queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id == 0) queue.push(source);
  return;
}

__global__ void bfs_kernel(int m, const uint64_t *row_offsets, 
                           const IndexT *column_indices, 
                           DistT *dists, Worklist2 in_queue, 
                           Worklist2 out_queue) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int src;
	if (in_queue.pop_id(tid, src)) {
		int row_begin = row_offsets[src];
		int row_end = row_offsets[src+1];
		for (int offset = row_begin; offset < row_end; ++ offset) {
			int dst = column_indices[offset];
			if ((dists[dst] == MYINFINITY) && 
          (atomicCAS(&dists[dst], MYINFINITY, dists[src]+1) == MYINFINITY)) {
				assert(out_queue.push(dst));
			}
		}
	}
}

class BfsLinearSideTask final : public BubbleBanditTask {
 private:
  bool with_profiler_;
  std::string file_type_;
  std::string graph_prefix_;
  std::string symmetrize_;
  std::string reverse_;
  std::string source_id_;
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

  auto get_current_time_in_micro() -> double {
    // Get the current time point
    auto now = std::chrono::high_resolution_clock::now();

    // Convert time point to microseconds
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    // Output the current time in microseconds
    return double(microseconds);
  }

  auto do_i_have_enough_time() -> bool {
    // Get the current time in microseconds
    auto current_time = get_current_time_in_micro();

    // Check if the current time is less than the end time
    return end_time_ - current_time > duration_;
  
  }

 public:
  BfsLinearSideTask(int task_id, std::string name, std::string device, std::string scheduler_addr, bool with_profiler, 
  std::string file_type, std::string graph_prefix, std::string symmetrize, std::string reverse, std::string source_id) 
  : BubbleBanditTask(task_id, name, device, scheduler_addr) {
    with_profiler_ = with_profiler;
    file_type_ = file_type;
    graph_prefix_ = graph_prefix;
    symmetrize_ = symmetrize;
    reverse_ = reverse;
    source_id_ = source_id;
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

  int64_t init(int64_t task_id) {
    assert(task_id == task_id_);
    std::cout << "Init task " << task_id << std::endl;
    init_event_ = true;
    return 0;
  }

  int64_t start(int64_t task_id, double end_time) {
    assert(task_id == task_id_);
    std::cout << "Start task " << task_id << " with end time " << end_time << std::endl;
    end_time_ = end_time;
    start_event_ = true;
    return 0;
  }

  int64_t pause(int64_t task_id) {
    assert(task_id == task_id_);
    std::cout << "Pause task " << task_id << std::endl;
    pause_event_ = true;
    return 0;
  }

  int64_t stop(int64_t task_id) {
    assert(task_id == task_id_);
    std::cout << "Stop task " << task_id << std::endl;
    stop_event_ = true;
    runner_.join();
    std::cout << "Task " << task_id << " stopped" << std::endl;
    kill(getpid(), SIGINT);
    return 0;
  }

  int64_t preempt(int64_t task_id) {
    assert(task_id == task_id_);
    std::cout << "Preempt task " << task_id << std::endl;
    preempt_event_ = true;
    return 0;
  }


  void run() {

    if (with_profiler_) {
      // TODO: You probably want to add some profiling logic here.
    }
    Graph g(graph_prefix_, file_type_, std::stoi(symmetrize_), std::stoi(reverse_));
    int source = std::stoi(source_id_);
    auto m = g.V();
    std::vector<DistT> distances(m, MYINFINITY);
    auto h_dists = &distances[0];

    auto state = BubbleBanditTask::State::CREATED;

    auto nnz = g.E();
    auto h_row_offsets = g.out_rowptr();
    auto h_column_indices = g.out_colidx();	
    uint64_t *d_row_offsets;
    VertexId *d_column_indices;
    DistT zero = 0;
    DistT * d_dists;
    Worklist2 queue1(m);
    Worklist2 queue2(m);
    Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
    int iter = 0;
    int nitems = 1;
    int nthreads = BLOCK_SIZE;
    int nblocks = (m - 1) / nthreads + 1;

    while (true) {
      switch (state) {
        case BubbleBanditTask::State::CREATED: {
          if (init_event_) {
            init_event_ = false;
            cudaSetDevice(device_.at(5) - '0');

            // auto m = g.V();
            // auto nnz = g.E();
            // auto h_row_offsets = g.out_rowptr();
            // auto h_column_indices = g.out_colidx();	
            // //print_device_info(0);
            // uint64_t *d_row_offsets;
            // VertexId *d_column_indices;

            CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(uint64_t)));
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(VertexId)));
            CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));

            // DistT zero = 0;
            // DistT * d_dists;
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_dists, m * sizeof(DistT)));
            CUDA_SAFE_CALL(cudaMemcpy(d_dists, h_dists, m * sizeof(DistT), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(&d_dists[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaDeviceSynchronize());

            // Worklist2 queue1(m);
            // Worklist2 queue2(m);
            // Worklist2 *in_frontier = &queue1, *out_frontier = &queue2;
            // int iter = 0;
            // int nitems = 1;
            // int nthreads = BLOCK_SIZE;
            // int nblocks = (m - 1) / nthreads + 1;
            printf("Launching CUDA BFS solver (%d threads/CTA) ...\n", nthreads);

            insert<<<1, nthreads>>>(source, *in_frontier);
            nitems = in_frontier->nitems();
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
            CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_dists, m * sizeof(DistT), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaFree(d_row_offsets));
            CUDA_SAFE_CALL(cudaFree(d_column_indices));
            CUDA_SAFE_CALL(cudaFree(d_dists));

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
            CUDA_SAFE_CALL(cudaMemcpy(h_dists, d_dists, m * sizeof(DistT), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaFree(d_row_offsets));
            CUDA_SAFE_CALL(cudaFree(d_column_indices));
            CUDA_SAFE_CALL(cudaFree(d_dists));
            state = BubbleBanditTask::State::CREATED;
            std::cout << "State from RUNNING to CREATED" << std::endl;
          } else {
            if (!do_i_have_enough_time()) {
              auto end_time = end_time_.load();
              if (end_time - get_current_time_in_micro() > 1000) {
                usleep((end_time - get_current_time_in_micro()) / 1000);
              }
            } else {
              ++ iter;
              nblocks = (nitems - 1) / nthreads + 1;
              std::cout << "iteration " << iter << ": frontier_size = " << nitems << std::endl;
              bfs_kernel <<<nblocks, nthreads>>> (m, d_row_offsets, d_column_indices, 
                  d_dists, *in_frontier, *out_frontier);
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              nitems = out_frontier->nitems();
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              Worklist2 *tmp = in_frontier;
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              in_frontier = out_frontier;
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              out_frontier = tmp;
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              out_frontier->reset();
              std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
              CUDA_SAFE_CALL(cudaDeviceSynchronize());
              if (nitems <= 0) {
                // TODO: clean up.
                goto BREAK_LOOP;
              }
            }
          }
          break;
        }
        case BubbleBanditTask::State::SUBMITTED: {
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
      // TODO: Jiashu: Add scheduler_client for finishing task.
      scheduler_client_.finish_task(task_id_);
    }
  }

  void finish() {
  }

  void start_runner() {
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
  program.add_argument("--reverse");
  program.add_argument("--source_id");

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
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
  auto reverse = program.get<std::string>("--reverse");
  auto source_id = program.get<std::string>("--source_id");

  auto task = BfsLinearSideTask(task_id, name, device, scheduler_addr, with_profiler, 
  file_type, graph_prefix, symmetrize, reverse, source_id);

  // task.init(task_id);
  // task.run();
  // task.stop(task_id);
  auto service = TaskServiceImpl(&task);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  server_ptr = server.get();
  std::cout << "Server listening on " << addr << std::endl;
  task.start_runner();
  server->Wait();

  signal(SIGINT, signalHandler);

  return 0;
}
