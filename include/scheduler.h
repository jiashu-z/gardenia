#ifndef BUBBLEBANDIT_INCLUDE_SCHEDULER_H
#define BUBBLEBANDIT_INCLUDE_SCHEDULER_H

#include "scheduler.grpc.pb.h"
#include "scheduler.pb.h"
#include <memory>
#include <grpcpp/create_channel.h>
#include <grpc/grpc.h>
#include <cassert>

class BubbleBanditSchedulerClient {
 private:
  const std::string scheduler_addr_;
  std::unique_ptr<Scheduler::Stub> stub_;
 public:
  BubbleBanditSchedulerClient(const std::string &scheduler_addr) : scheduler_addr_(scheduler_addr) {
    auto channel = grpc::CreateChannel(scheduler_addr_, grpc::InsecureChannelCredentials());
    stub_ = Scheduler::NewStub(channel);
  }

  auto finish_task(const int task_id) const -> int {
    grpc::ClientContext context;
    FinishTaskArgs args;
    FinishTaskReply reply;
    args.set_task_id(task_id);
    auto status = stub_->FinishTask(&context, args, &reply);
    if (status.ok()) {
      assert(reply.status() == 0);
      return reply.status();
    } else {
      return -1;
    }
  }
};

#endif