#ifndef BUBBLEBANDIT_INCLUDE_TASK_H
#define BUBBLEBANDIT_INCLUDE_TASK_H

#include "task.grpc.pb.h"
#include "task.pb.h"
#include <grpcpp/impl/channel_interface.h>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <cassert>
#include "scheduler.h"
#include <string>
#include <thread>
#include <cassert>
#include <map>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <signal.h>

class BubbleBanditTask {
 protected:
  const int64_t task_id_;
  const std::string name_;
  const std::string device_;
  const std::string scheduler_addr_;
  const BubbleBanditSchedulerClient scheduler_client_;

 public:

  enum State {
    SUBMITTED,
    CREATED,
    PAUSED,
    RUNNING,
    STOPPED,
    FINISHED,
  };

  std::atomic<bool> init_event_;
  std::atomic<bool> start_event_;
  std::atomic<bool> pause_event_;
  std::atomic<bool> stop_event_;
  std::atomic<bool> preempt_event_;
  std::thread runner_;
  double duration_;
  std::atomic<double> end_time_;
  int profiler_level_;
  std::map<std::string, std::vector<double>> time_records_;
  State state_;

  virtual auto submitted_to_created() -> void = 0;

  virtual auto created_to_paused() -> void = 0;

  virtual auto paused_to_running() -> void = 0;

  virtual auto running_to_paused() -> void = 0;

  virtual auto running_to_finished() -> void = 0;

  virtual auto to_stopped() -> void = 0;

  virtual auto is_finished() -> bool = 0;

  virtual auto step() -> void = 0;

  auto get_current_time_in_micro() -> double {
    // Get the current time point
    auto now = std::chrono::high_resolution_clock::now();

    // Convert time point to microseconds
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

    // Output the current time in microseconds
    return static_cast<double>(microseconds) / 1000000.0;
  }

  auto do_i_have_enough_time() -> bool {
    auto current_time = get_current_time_in_micro();

    return end_time_ - current_time > duration_;
  }

  auto record_time(const std::string &action) -> void {
    if (time_records_.find(action) != time_records_.end()) {
      time_records_[action].push_back(get_current_time_in_micro());
    } else {
      printf("Creating new action %s\n", action.c_str());
      auto v = std::vector<double>();
      v.push_back(get_current_time_in_micro());
      time_records_.insert(std::make_pair(action, v));
    }
  }

  BubbleBanditTask(int64_t task_id,
                   std::string name,
                   std::string device,
                   std::string scheduler_addr,
                   int profiler_level) :
      task_id_(task_id),
      name_(name),
      device_(device),
      scheduler_addr_(scheduler_addr),
      scheduler_client_(scheduler_addr) {
    init_event_ = false;
    start_event_ = false;
    pause_event_ = false;
    stop_event_ = false;
    preempt_event_ = false;
    profiler_level_ = profiler_level;
    state_ = SUBMITTED;
    end_time_ = -1;
  }

  virtual int64_t init(int64_t task_id) {
    assert(task_id == task_id_);
    printf("Init task %ld\n", task_id_);
    init_event_ = true;
    return 0;
  }

  virtual int64_t start(int64_t task_id, double end_time) {
    assert(task_id == task_id_);
    printf("Start task %ld\n", task_id_);
    start_event_ = true;
    end_time_ = end_time;
    return 0;
  }

  virtual int64_t pause(int64_t task_id) {
    assert(task_id == task_id_);
    printf("Pause task %ld\n", task_id_);
    pause_event_ = true;
    return 0;
  }

  virtual int64_t stop(int64_t task_id) {
    assert(task_id == task_id_);
    printf("Stop task %ld\n", task_id_);
    stop_event_ = true;
    runner_.join();
    kill(getpid(), SIGINT);
    return 0;
  }

  virtual int64_t preempt(int64_t task_id) {
    assert(task_id == task_id_);
    printf("Preempt task %ld\n", task_id_);
    preempt_event_ = true;
    return 0;
  }

  virtual void finish() {
    auto ret = scheduler_client_.finish_task(task_id_);
    assert(ret == 0);
    printf("Finish task %ld\n", task_id_);
  }

  virtual void run() {
    record_time("SUBMITTED_TO_CREATED_START");
    submitted_to_created();
    state_ = CREATED;
    record_time("SUBMITTED_TO_CREATED_END");
    printf("State from SUBMITTED to CREATED\n");

    while (true) {
      switch (state_) {
        case CREATED: {
          if (init_event_) {
            init_event_ = false;
            record_time("CREATED_TO_PAUSED_START");
            created_to_paused();
            state_ = PAUSED;
            record_time("CREATED_TO_PAUSED_END");
            printf("State from CREATED to PAUSED\n");
          }
          break;
        }
        case PAUSED: {
          if (start_event_) {
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            start_event_ = false;
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            record_time("PAUSED_TO_RUNNING_START");
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            paused_to_running();
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            state_ = RUNNING;
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            record_time("PAUSED_TO_RUNNING_END");
            std::cout << __FILE__ << ":" << __LINE__ << std::endl;
            printf("State from PAUSED to RUNNING\n");
          }
          break;
        }
        case RUNNING: {
          if (pause_event_) {
            pause_event_ = false;
            record_time("RUNNING_TO_PAUSED_START");
            running_to_paused();
            state_ = PAUSED;
            record_time("RUNNING_TO_PAUSED_END");
            printf("State from RUNNING to PAUSED\n");
          }
          if (is_finished()) {
            record_time("RUNNING_TO_FINISHED_START");
            running_to_finished();
            state_ = FINISHED;
            record_time("RUNNING_TO_FINISHED_END");
            printf("State from RUNNING to FINISHED\n");
            goto LOOP_END;
          }
          if (do_i_have_enough_time()) {
            record_time("RUNNING_STEP_START");
            step();
            record_time("RUNNING_STEP_END");
          } else {
            record_time("SLEEPING_STEP_START");
            if (end_time_.load() - get_current_time_in_micro() > 0.001) {
              usleep((end_time_.load() - get_current_time_in_micro()) * 1000000);
            }
            record_time("SLEEPING_STEP_END");
          }
          break;
        }
        default: {
          assert(false);
        }
      }
      if (stop_event_) {
        record_time("TO_STOPPED_START");
        to_stopped();
        state_ = STOPPED;
        record_time("TO_STOPPED_END");
        printf("State to STOPPED\n");
        goto LOOP_END;
      }
      usleep(10000);
    }

    LOOP_END:

    std::ofstream out_file;
    std::string out_file_name = name_ + "_time_profile_" + std::to_string(task_id_) + ".txt";
    out_file.precision(9);
    out_file << std::fixed;
    out_file.open(out_file_name);
    assert(out_file.is_open());
    for (auto &kv : time_records_) {
      out_file << kv.first << ",";
      for (auto &v : kv.second) {
        out_file << v << ",";
      }
      out_file << std::endl;
    }
    out_file.flush();
    out_file.close();

    if (stop_event_) {
      stop_event_ = false;
    } else {
      finish();
    }

    printf("Task %ld finished\n", task_id_);
  }

  virtual void start_runner() {
    printf("Start runner of task %ld\n", task_id_);
    runner_ = std::thread([this] { run(); });
  }

};

class TaskServiceImpl final : public Task::Service {
 private:
  BubbleBanditTask *const task_;
 public:
  TaskServiceImpl(BubbleBanditTask *task) : task_(task) {
  }

  grpc::Status InitTask(grpc::ServerContext *context, const InitTaskArgs *request, InitTaskReply *response) override {
    auto task_id = request->task_id();
    auto status = task_->init(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status StartTask(grpc::ServerContext *context,
                         const StartTaskArgs *request,
                         StartTaskReply *response) override {
    auto task_id = request->task_id();
    auto end_time = request->end_time();
    printf("StartTask task_id: %ld, end_time: %f\n", task_id, end_time);
    auto status = task_->start(task_id, end_time);
    printf("StartTask task_id: %ld, end_time: %f, status: %ld\n", task_id, end_time, status);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status PauseTask(grpc::ServerContext *context,
                         const PauseTaskArgs *request,
                         PauseTaskReply *response) override {
    auto task_id = request->task_id();
    auto status = task_->pause(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status StopTask(grpc::ServerContext *context, const StopTaskArgs *request, StopTaskReply *response) override {
    auto task_id = request->task_id();
    auto status = task_->stop(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status PreemptTask(grpc::ServerContext *context,
                           const PreemptTaskArgs *request,
                           PreemptTaskReply *response) override {
    auto task_id = request->task_id();
    auto status = task_->preempt(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }
};

class TaskClient {
 private:
  std::string addr_;
  std::unique_ptr<Task::Stub> stub_;

 public:
  TaskClient(const std::string addr) {
    addr_ = addr;
    auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    stub_ = Task::NewStub(channel);
  }

  int64_t init_task(int64_t task_id) {
    grpc::ClientContext context;
    InitTaskArgs args;
    args.set_task_id(task_id);
    InitTaskReply reply;
    auto status = stub_->InitTask(&context, args, &reply);
    assert(status.ok());
    assert (reply.status() == 0);
    return reply.status();
  }

  int64_t start_task(int64_t task_id) {
    grpc::ClientContext context;
    StartTaskArgs args;
    args.set_task_id(task_id);
    StartTaskReply reply;
    auto status = stub_->StartTask(&context, args, &reply);
    assert(status.ok());
    assert (reply.status() == 0);
    return reply.status();
  }

  int64_t pause_task(int64_t task_id) {
    grpc::ClientContext context;
    PauseTaskArgs args;
    args.set_task_id(task_id);
    PauseTaskReply reply;
    auto status = stub_->PauseTask(&context, args, &reply);
    assert(status.ok());
    assert (reply.status() == 0);
    return reply.status();
  }

  int64_t stop_task(int64_t task_id) {
    grpc::ClientContext context;
    StopTaskArgs args;
    args.set_task_id(task_id);
    StopTaskReply reply;
    auto status = stub_->StopTask(&context, args, &reply);
    assert(status.ok());
    assert (reply.status() == 0);
    return reply.status();
  }

  int64_t preempt_task(int64_t task_id) {
    grpc::ClientContext context;
    PreemptTaskArgs args;
    args.set_task_id(task_id);
    PreemptTaskReply reply;
    auto status = stub_->PreemptTask(&context, args, &reply);
    assert(status.ok());
    assert (reply.status() == 0);
    return reply.status();
  }

};

#endif
