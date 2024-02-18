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

class BubbleBanditTask {
 protected:
  const int64_t task_id_;
  const std::string name_;
  const std::string device_;
  const std::string scheduler_addr_;

 public:
  BubbleBanditTask(int64_t task_id, std::string name, std::string device, std::string scheduler_addr) : 
  task_id_(task_id), name_(name), device_(device), scheduler_addr_(scheduler_addr) {
  }

  virtual int64_t init(int64_t task_id) = 0;

  virtual int64_t start(int64_t task_id) = 0;

  virtual int64_t pause(int64_t task_id) = 0;

  virtual int64_t stop(int64_t task_id) = 0;

  virtual int64_t preempt(int64_t task_id) = 0;

  virtual void run() = 0;

  virtual void finish() = 0;

  virtual void start_runner() = 0;

  enum State {
    SUBMITTED,
    CREATED,
    PENDING,
    RUNNING,
  };
};

class TaskServiceImpl final : public Task::Service {
 private:
  BubbleBanditTask * const task_;
 public:
  TaskServiceImpl(BubbleBanditTask *task) : task_(task) {
  }

  grpc::Status InitTask(grpc::ServerContext* context, const InitTaskArgs* request, InitTaskReply* response) override {
    auto task_id = request->task_id();
    auto status = task_->init(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status StartTask(grpc::ServerContext* context, const StartTaskArgs* request, StartTaskReply* response) override {
    auto task_id = request->task_id();
    auto status = task_->start(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status PauseTask(grpc::ServerContext* context, const PauseTaskArgs* request, PauseTaskReply* response) override {
    auto task_id = request->task_id();
    auto status = task_->pause(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status StopTask(grpc::ServerContext* context, const StopTaskArgs* request, StopTaskReply* response) override {
    auto task_id = request->task_id();
    auto status = task_->stop(task_id);
    if (status == 0) {
      return grpc::Status::OK;
    } else {
      return grpc::Status::CANCELLED;
    }
  }

  grpc::Status PreemptTask(grpc::ServerContext* context, const PreemptTaskArgs* request, PreemptTaskReply* response) override {
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