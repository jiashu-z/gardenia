#include "bfs.h"
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

__global__ void insert(int source, Worklist2 queue) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id == 0) queue.push(source);
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
    int row_end = row_offsets[src + 1];
    for (int offset = row_begin; offset < row_end; ++offset) {
      int dst = column_indices[offset];
      if ((dists[dst] == MYINFINITY) &&
          (atomicCAS(&dists[dst], MYINFINITY, dists[src] + 1) == MYINFINITY)) {
        assert(out_queue.push(dst));
      }
    }
  }
}

class BfsLinearSideTask final : public BubbleBanditTask {
private:
    std::string file_type_;
    std::string graph_prefix_;
    std::string symmetrize_;
    std::string reverse_;
    std::string source_id_;
    int max_iter_;
    
    Graph *g_ptr;
    VertexId m;
    size_t nnz;
    uint64_t *h_row_offsets;
    VertexId *h_column_indices;
    uint64_t *d_row_offsets;
    VertexId *d_column_indices;
    DistT *h_dists;
    
    int iter;
    int item_num;
    int thread_num;
    int block_num;
    int source;
    Worklist2 queue1;
    Worklist2 queue2;
    Worklist2 *in_frontier;
    Worklist2 *out_frontier;
    
    DistT zero = 0;
    DistT *d_dists;

public:
    BfsLinearSideTask(int64_t task_id, std::string name, std::string device, std::string scheduler_addr, double duration, int profiler_level,
                      std::string file_type, std::string graph_prefix, std::string symmetrize, std::string reverse,
                      std::string source_id, int max_iter)
            : BubbleBanditTask(task_id, name, device, scheduler_addr, profiler_level), queue1(0), queue2(0) {
      file_type_ = file_type;
      graph_prefix_ = graph_prefix;
      symmetrize_ = symmetrize;
      reverse_ = reverse;
      source_id_ = source_id;
      duration_ = duration;
      max_iter_ = max_iter;
    }
    
    auto submitted_to_created() -> void override {
      auto device = device_.at(5) - '0';
      std::cout << "Device: " << device << std::endl;
      cudaSetDevice(device);
      g_ptr = new Graph(graph_prefix_, file_type_, std::stoi(symmetrize_), 1);
      auto &g = *g_ptr;
      
      source = std::stoi(source_id_);
      m = g.V();
      std::vector<DistT> distances(m, MYINFINITY);
      h_dists = &distances[0];
      
      
      nnz = g.E();
      h_row_offsets = g.out_rowptr();
      h_column_indices = g.out_colidx();
      
      zero = 0;
      std::cout << "Max size: " << m << std::endl;
      queue1 = Worklist2(m);
      queue2 = Worklist2(m);

      in_frontier = &queue1;
      out_frontier = &queue2;

      iter = 0;
      item_num = 1;
      thread_num = BLOCK_SIZE;
      block_num = (m - 1) / thread_num + 1;
      
      insert<<<1, thread_num>>>(source, *in_frontier);
      item_num = in_frontier->nitems();

      printf("Launching CUDA BFS solver (%d threads/CTA) ...\n", thread_num);

      state_ = BubbleBanditTask::State::CREATED;
    }
    
    auto created_to_paused() -> void override {
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      
      CUDA_SAFE_CALL(cudaMalloc((void **) &d_row_offsets, (m + 1) * sizeof(uint64_t)));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaMalloc((void **) &d_column_indices, nnz * sizeof(VertexId)));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(VertexId), cudaMemcpyHostToDevice));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      
      CUDA_SAFE_CALL(cudaMalloc((void **) &d_dists, m * sizeof(DistT)));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaMemcpy(d_dists, h_dists, m * sizeof(DistT), cudaMemcpyHostToDevice));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaMemcpy(&d_dists[source], &zero, sizeof(zero), cudaMemcpyHostToDevice));
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      std::cout << __FILE__ << ":" << __LINE__ << std::endl;
    }
    
    auto step() -> void override {
      ++ iter;
      block_num = (item_num - 1) / thread_num + 1;
      std::cout << "iteration " << iter << ": frontier_size = " << item_num << std::endl;
      bfs_kernel <<<block_num, thread_num>>> (m, d_row_offsets, d_column_indices, 
          d_dists, *in_frontier, *out_frontier);
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      // CudaTest("solving bfs_kernel failed");
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      item_num = out_frontier->nitems();
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      Worklist2 *tmp = in_frontier;
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      in_frontier = out_frontier;
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      out_frontier = tmp;
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
      out_frontier->reset();
      std::cout << __FILE__ << ": "<< __LINE__ << std::endl;
//       ++iter;
//       block_num = (item_num - 1) / thread_num + 1;
//       std::cout << "iteration " << iter << ": frontier_size = " << item_num <<
//                 std::endl;
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
//       bfs_kernel<<<block_num, thread_num>>>(m, d_row_offsets, d_column_indices,
//                                             d_dists, *in_frontier, *out_frontier
//       );
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
// //              CUDA_SAFE_CALL(cudaDeviceSynchronize())
//       item_num = out_frontier->nitems();
//       // PRINT ITEM NUM
//       std::cout << "ITEM_NUM = " << item_num << std::endl;
//       std::cout << "New frontier_size = " << item_num <<
//                 std::endl;
// //              CUDA_SAFE_CALL(cudaDeviceSynchronize())
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
//       Worklist2 *tmp = in_frontier;
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
//       in_frontier = out_frontier;
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
//       out_frontier = tmp;
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
//       out_frontier->
              
//               reset();
      
//       std::cout << __FILE__ << ": " << __LINE__ <<
//                 std::endl;
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
  argparse::ArgumentParser program("program_name");
  program.add_argument("-n", "--name");
  program.add_argument("-s", "--scheduler_addr");
  program.add_argument("-i", "--task_id");
  program.add_argument("-d", "--device");
  program.add_argument("-a", "--addr");
  program.add_argument("--duration");
  // TODO: Jiashu: Fix profiler flag
  program.add_argument("-p", "--profiler_level");
  program.add_argument("-t", "--file_type");
  program.add_argument("-g", "--graph_prefix");
  program.add_argument("--symmetrize");
  program.add_argument("--reverse");
  program.add_argument("--source_id");
  program.add_argument("--max_iter");
  
  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }
  
  auto name = program.get<std::string>("--name");
  auto scheduler_addr = program.get<std::string>("--scheduler_addr");
  auto task_id = std::stoi(program.get<std::string>("--task_id"));
  auto device = program.get<std::string>("--device");
  auto addr = program.get<std::string>("--addr");
  auto duration = std::stod(program.get<std::string>("--duration"));
  auto profiler_level = std::stoi(program.get<std::string>("--profiler_level"));
  auto file_type = program.get<std::string>("--file_type");
  auto graph_prefix = program.get<std::string>("--graph_prefix");
  auto symmetrize = program.get<std::string>("--symmetrize");
  auto reverse = program.get<std::string>("--reverse");
  auto source_id = program.get<std::string>("--source_id");
  auto max_iter = std::stoi(program.get<std::string>("--max_iter"));
  
  auto task = BfsLinearSideTask(task_id, name, device, scheduler_addr, duration, profiler_level,
                                file_type, graph_prefix, symmetrize, reverse, source_id, max_iter);
  
  // task.init(task_id);
  // task.run();
  // task.stop(task_id);
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
