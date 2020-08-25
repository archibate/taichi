#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Demote Operations into pieces for backends to deal easier
class DemoteOperations : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;

  explicit AlgSimp(bool fast_math_)
      : BasicStmtVisitor(), fast_math(fast_math_) {
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs;
    auto rhs = stmt->rhs;
    if (stmt->op_type == BinaryOpType::floordiv) {
      if (is_integral(rhs->element_type()) && is_integral(lhs->element_type())) {
        // @ti.func
        // def ifloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     if (a < 0) != (b < 0) and a and b * r != a:
        //         r = r - 1
        //     return r
        auto ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::div, lhs, rhs);
        auto zero = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(0));
        auto lhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, lhs, zero.get());
        auto rhs_ltz = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_lt, rhs, zero.get());
        auto rhs_mul_ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::mul, rhs, ret.get());
        auto cond1 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_ne, lhs_ltz.get(), rhs_ltz.get());
        auto cond2 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, lhs, zero.get());
        auto cond3 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::cmp_eq, rhs_mul_ret.get(), lhs);
        auto cond12 = Stmt::make<BinaryOpStmt>(
            BinaryOpType::bit_and, cond1.get(), cond2.get());
        auto cond = Stmt::make<BinaryOpStmt>(
            BinaryOpType::bit_and, cond12.get(), cond3.get());
        auto real_ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::add, ret.get(), cond.get());

        modifier.insert_before(stmt, std::move(ret));
        modifier.insert_before(stmt, std::move(zero));
        modifier.insert_before(stmt, std::move(lhs_ltz));
        modifier.insert_before(stmt, std::move(rhs_ltz));
        modifier.insert_before(stmt, std::move(rhs_mul_ret));
        modifier.insert_before(stmt, std::move(cond1));
        modifier.insert_before(stmt, std::move(cond2));
        modifier.insert_before(stmt, std::move(cond3));
        modifier.insert_before(stmt, std::move(cond12));
        modifier.insert_before(stmt, std::move(cond));
        modifier.insert_before(stmt, std::move(real_ret));
        modifier.erase(stmt);
      } else {
        // @ti.func
        // def ffloordiv(a, b):
        //     r = ti.raw_div(a, b)
        //     return ti.floor(r)
        auto ret = Stmt::make<BinaryOpStmt>(
            BinaryOpType::div, lhs, rhs);
        auto floor = Stmt::make<UnaryOpStmt>(
            UnaryOpType::floor, ret.get());
        modifier.insert_before(stmt, std::move(ret));
        modifier.insert_before(stmt, std::move(floor));
        modifier.erase(stmt);
      }
    }
  }
};

namespace irpass {

bool demote_operations(IRNode *root) {
  TI_AUTO_PROF;
  return DemoteOperations::run(root);
}

}  // namespace irpass
