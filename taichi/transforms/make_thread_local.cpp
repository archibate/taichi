#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace {

bool is_atomic_op_linear(AtomicOpType op_type) {
  return op_type == AtomicOpType::add || op_type == AtomicOpType::sub;
}

void make_thread_local_offload(OffloadedStmt *offload) {
  // TODO: deal with struct for
  if (offload->task_type != offload->range_for)
    return;

  // Gather all atomic adds/subs destinations
  // We use std::vector instead of std::set to keep an deterministic order here.
  std::vector<GlobalPtrStmt *> atomic_destinations;
  // TODO: this is again an abuse since it gathers nothing. Need to design a IR
  // map/reduce system
  auto linear_atomics =
      irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
        if (auto atomic_op = stmt->cast<AtomicOpStmt>()) {
          if (is_atomic_op_linear(atomic_op->op_type)) {
            // Local or global tmp atomics does not count
            if (auto dest = atomic_op->dest->cast<GlobalPtrStmt>()) {
              if (std::find(atomic_destinations.begin(),
                            atomic_destinations.end(),
                            dest) == atomic_destinations.end()) {
                atomic_destinations.push_back(dest);
              }
            }
          }
        }
        return false;
      });

  std::vector<GlobalPtrStmt *> valid_reduction_values;

  for (auto dest : atomic_destinations) {
    // check if there is any other global load/store/atomic operations
    auto related_global_mem_ops =
        irpass::analysis::gather_statements(offload, [&](Stmt *stmt) {
          if (auto load = stmt->cast<GlobalLoadStmt>()) {
            if (maybe_same_address(load->ptr, dest)) {
              return true;
            }
          } else if (auto store = stmt->cast<GlobalStoreStmt>()) {
            if (maybe_same_address(store->ptr, dest)) {
              return true;
            }
          } else if (auto atomic = stmt->cast<AtomicOpStmt>()) {
            if (maybe_same_address(atomic->dest, dest)) {
              return !is_atomic_op_linear(atomic->op_type);
            }
          }
          for (auto &op : stmt->get_operands()) {
            // Make sure the values of related atomic add operation are not
            // used.
            if (auto atomic = op->cast<AtomicOpStmt>()) {
              if (maybe_same_address(atomic->dest, dest)) {
                return true;
              }
            }
          }
          return false;  // Now we are sure the statement is not related to the
                         // destination
        });
    TI_ASSERT(dest->width() == 1);
    // We can only optimized reductions to global ptrs with form like loss[None]
    // (0-D tensors) for now
    if (related_global_mem_ops.empty() &&
        dest->snodes[0]->type == SNodeType::place && dest->indices.empty()) {
      valid_reduction_values.push_back(dest);
    }
  }

  // We use 8 here (instead of sizeof(data_type) for TLS variable alignment
  // (sizeof(f64) = 8).
  constexpr std::size_t tls_stride = 8;

  std::size_t tls_offset = 0;

  for (auto dest : valid_reduction_values) {
    auto data_type = dest->ret_type.data_type;
    // Step 1:
    // Create thread local storage
    {
      if (offload->prologue == nullptr) {
        offload->prologue = std::make_unique<Block>();
      }
      auto tls_ptr = offload->prologue->push_back<ThreadLocalPtrStmt>(
          tls_offset, VectorType(1, data_type));
      auto zero = offload->prologue->insert(
          std::make_unique<ConstStmt>(TypedConstant(data_type, 0)), -1);
      // Zero-fill
      // TODO: do not use GlobalStore for TLS ptr.
      offload->prologue->push_back<GlobalStoreStmt>(tls_ptr, zero);
    }

    // Step 2:
    // Make loop body accumulate to TLS ptr instead of global ptr
    {
      auto tls_ptr = offload->body->insert(
          Stmt::make<ThreadLocalPtrStmt>(tls_offset, VectorType(1, data_type)),
          0);
      dest->replace_with(tls_ptr);
    }

    // Step 3:
    // Atomic-add thread local contribution to its global version
    {
      if (offload->epilogue == nullptr) {
        offload->epilogue = std::make_unique<Block>();
      }
      auto tls_ptr = offload->epilogue->push_back<ThreadLocalPtrStmt>(
          tls_offset, VectorType(1, data_type));
      // TODO: do not use global load from TLS.
      auto tls_load = offload->epilogue->push_back<GlobalLoadStmt>(tls_ptr);
      auto global_ptr = offload->epilogue->insert(
          std::unique_ptr<Stmt>(
              (Stmt *)irpass::analysis::clone(dest).release()),
          -1);
      offload->epilogue->push_back<AtomicOpStmt>(AtomicOpType::add, global_ptr,
                                                 tls_load);
    }
    tls_offset += tls_stride;
    if (tls_offset >= taichi_tls_buffer_size)
      break;  // Do not overflow TLS buffer.  TODO: make it adaptive
  }
}

}  // namespace

namespace irpass {

// This pass should happen after offloading but before lower_access
void make_thread_local(IRNode *root) {
  TI_AUTO_PROF;
  auto root_block = root->cast<Block>();
  TI_ASSERT(root_block);
  for (auto &offload : root_block->statements) {
    make_thread_local_offload(offload->cast<OffloadedStmt>());
  }
  typecheck(root);
  fix_block_parents(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
