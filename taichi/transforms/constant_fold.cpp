#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/ir/snode.h"
#include <deque>
#include <set>
#include <cmath>

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

  struct JITEvaluatorId
  {
    int op;
    DataType ret, lhs, rhs;
    bool is_binary;

    explicit operator int() const // make STL map happy
    {
      return (int)op | (int)!!is_binary << 7
        | (int) ret << 8 | (int) lhs << 16 | (int) rhs << 24;
    }

    UnaryOpType unary_op() const
    {
      return (UnaryOpType) op;
    }

    BinaryOpType binary_op() const
    {
      return (BinaryOpType) op;
    }

#if 0
    bool operator==(const JITEvaluatorId &b)
    {
      return op == b.op && ret == b.ret && lhs == b.lhs && rhs == b.rhs
          && is_binary == b.is_binary;
    }

    struct hash {
      size_t operator()(const JITEvaluatorId &id)
      {
        hash<int> hop;
        hash<DataType> hdt;
        return hop(op | is_binary << 7)
          ^ hdt(ret) ^ hdt(lhs) ^ hdt(rhs);
      }
    };
#endif
  };

  static Kernel *get_jit_evaluator_kernel(JITEvaluatorId const &id)
  {
    auto &cache = get_current_program().jit_evaluator_cache;
    int iid = int(id);
    auto it = cache.find(iid);
    if (it != cache.end()) // cached?
      return it->second.get();
    static int jic = 0; // X: race?
    auto kernel_name = fmt::format("jit_evaluator_{}", jic++);
    auto func = [&] () {
      //insert_die_loop();
      auto lhstmt = Stmt::make<ArgLoadStmt>(0, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(1, false);
      pStmt oper;
      if (id.is_binary) {
        oper = Stmt::make<BinaryOpStmt>(id.binary_op(), lhstmt.get(), rhstmt.get());
      } else {
        oper = Stmt::make<UnaryOpStmt>(id.unary_op(), lhstmt.get());
        if (unary_op_is_cast(id.unary_op())) {
          oper->cast<UnaryOpStmt>()->cast_type = id.rhs;
        }
      }
      auto ret = Stmt::make<KernelReturnStmt>(oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      if (id.is_binary)
        current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
      //insert_die("DEAD");
    };
    auto ker = std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_ret(id.ret);
    ker->insert_arg(id.lhs, false);
    if (id.is_binary)
      ker->insert_arg(id.rhs, false);
    auto *ker_ptr = ker.get();
    TI_TRACE("Saving JIT evaluator cache entry id={}", iid);
    cache[iid] = std::move(ker);
    return ker_ptr;
  }

  static bool is_good_type(DataType dt)
  {
      switch (dt) {
      case DataType::i32:
      case DataType::f32:
      case DataType::i64:
      case DataType::f64:
        return true;
      default:
        return false;
      }
  }

  static bool jit_from_binary_op(TypedConstant &ret, BinaryOpStmt *stmt,
      const TypedConstant &lhs, const TypedConstant &rhs)
  {
    // ConstStmt of `bad` types like `i8` is not supported by LLVM.
    // Dis: https://github.com/taichi-dev/taichi/pull/839#issuecomment-625902727
    if (!is_good_type(ret.dt))
      return false;
    JITEvaluatorId id{(int)stmt->op_type, ret.dt, lhs.dt, rhs.dt,
      true};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
    //TI_INFO("JITARGSf = {} {}", lhs.val_f32, rhs.val_f32);
    TI_INFO("JITARGSi = {} {}", lhs.val_i32, rhs.val_i32);
    ctx.set_arg<int64_t>(0, lhs.val_i64);
    ctx.set_arg<int64_t>(1, rhs.val_i64);
    irpass::print(ker->ir);
    (*ker)();
    ret.val_i64 = get_current_program().fetch_result<int64_t>(0);
    //TI_INFO("JITEVALf = {}", ret.val_f32);
    TI_INFO("JITEVALi = {}", ret.val_i32);
    return true;
  }

  static bool jit_from_unary_op(TypedConstant &ret, UnaryOpStmt *stmt,
      const TypedConstant &lhs)
  {
    // TODO: remove this:
    if (unary_op_is_cast(stmt->op_type))
      return false;
    if (!is_good_type(ret.dt))
      return false;
    JITEvaluatorId id{(int)stmt->op_type, ret.dt, lhs.dt, stmt->cast_type,
      false};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
    //TI_INFO("JITARGSf = {} {}", lhs.val_f32);
    TI_INFO("JITARGSi = {}", lhs.val_i32);
    ctx.set_arg<int64_t>(0, lhs.val_i64);
    irpass::print(ker->ir);
    (*ker)();
    ret.val_i64 = get_current_program().fetch_result<int64_t>(0);
    //TI_INFO("JITEVALf = {}", ret.val_f32);
    TI_INFO("JITEVALi = {}", ret.val_i32);
    return true;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    TI_INFO("JIT_ret_dt = {}", dst_type);
    if (jit_from_binary_op(new_constant, stmt, lhs->val[0], rhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    auto lhs = stmt->operand->cast<ConstStmt>();
    if (!lhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    TI_INFO("JIT_ret_dt = {}", dst_type);
    if (jit_from_unary_op(new_constant, stmt, lhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  static void run(IRNode *node) {
    ConstantFold folder;
    while (true) {
      bool modified = false;
      try {
        node->accept(&folder);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void constant_fold(IRNode *root) {
  // @archibate found that `debug=True` will cause JIT kernels
  // failed to evaluate correctly (always return 0), so we simply
  // disable constant_fold when config.debug is turned on.
  // Dis: https://github.com/taichi-dev/taichi/pull/839#issuecomment-626107010
  if (get_current_program().config.debug)
    return;
  return ConstantFold::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
