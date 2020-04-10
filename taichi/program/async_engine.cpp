#include "async_engine.h"

#include <memory>

#include "taichi/program/kernel.h"
#include "taichi/program/program.h"
#include "taichi/codegen/codegen_cpu.h"

TLANG_NAMESPACE_BEGIN

KernelLaunchRecord::KernelLaunchRecord(Context context,
                                       Kernel *kernel,
                                       OffloadedStmt *stmt)
    : context(context), kernel(kernel), stmt(stmt) {
}

uint64 ExecutionQueue::hash(OffloadedStmt *stmt) {
  // TODO: upgrade this using IR comparisons
  std::string serialized;
  irpass::print(stmt, &serialized);
  uint64 ret = 0;
  for (uint64 i = 0; i < serialized.size(); i++) {
    ret = ret * 100000007UL + (uint64)serialized[i];
  }
  return ret;
}

void ExecutionQueue::enqueue(KernelLaunchRecord ker) {
  task_queue.push_back(ker);
}

void ExecutionQueue::synchronize() {
  while (!task_queue.empty()) {
    auto ker = task_queue.front();
    std::string serialized;
    irpass::re_id(ker.stmt);
    irpass::print(ker.stmt);
    auto h = hash(ker.stmt);
    if (compiled_func.find(h) == compiled_func.end()) {
      compiled_func[h] = CodeGenCPU(ker.kernel, ker.stmt).codegen();
    }
    compiled_func[h](ker.context);
    task_queue.pop_front();
  }
}

void AsyncEngine::launch(Kernel *kernel) {
  if (!kernel->lowered)
    kernel->lower();
  auto block = dynamic_cast<Block *>(kernel->ir);
  TI_ASSERT(block);
  auto &offloads = block->statements;
  for (std::size_t i = 0; i < offloads.size(); i++) {
    auto offload = offloads[i]->as<OffloadedStmt>();
    task_queue.emplace_back(kernel->program.get_context(), kernel, offload);
  }
  optimize();
  synchronize();
}

void AsyncEngine::synchronize() {
  while (!task_queue.empty()) {
    queue.enqueue(task_queue.front());
    task_queue.pop_front();
  }
  queue.synchronize();
}

TLANG_NAMESPACE_END
