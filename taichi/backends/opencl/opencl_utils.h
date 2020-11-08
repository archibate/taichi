#pragma once

#include "taichi/lang_util.h"
#include "taichi/common/core.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

TLANG_NAMESPACE_BEGIN
namespace opencl {

// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
inline std::string opencl_data_type_name(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::i8))
    return "char";
  else if (dt->is_primitive(PrimitiveTypeID::i16))
    return "short";
  else if (dt->is_primitive(PrimitiveTypeID::i32))
    return "int";
  else if (dt->is_primitive(PrimitiveTypeID::i64))
    return "long";
  else if (dt->is_primitive(PrimitiveTypeID::u8))
    return "uchar";
  else if (dt->is_primitive(PrimitiveTypeID::u16))
    return "ushort";
  else if (dt->is_primitive(PrimitiveTypeID::u32))
    return "uint";
  else if (dt->is_primitive(PrimitiveTypeID::u64))
    return "ulong";
  else if (dt->is_primitive(PrimitiveTypeID::f16))
    return "half";
  else if (dt->is_primitive(PrimitiveTypeID::f32))
    return "float";
  else if (dt->is_primitive(PrimitiveTypeID::f64))
    return "double";
  else if (dt->is_primitive(PrimitiveTypeID::u1))
    return "bool";
  TI_ERROR("Unsupported DataType={} on OpenCL backend",
           data_type_name(dt));
}

inline std::string opencl_atomic_op_type_name(AtomicOpType op) {
  switch (op) {
    case AtomicOpType::add:
      return "add";
    case AtomicOpType::sub:
      return "sub";
    case AtomicOpType::bit_or:
      return "or";
    case AtomicOpType::bit_xor:
      return "xor";
    case AtomicOpType::bit_and:
      return "and";
    case AtomicOpType::max:
      return "max";
    case AtomicOpType::min:
      return "min";
    default:
      TI_ERROR("Unsupported AtomicOpType={} on OpenCL backend",
               atomic_op_type_name(op));
  }
}

inline bool opencl_is_binary_op_infix(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::max:
    case BinaryOpType::min:
    case BinaryOpType::atan2:
    case BinaryOpType::pow:
      return false;
    default:
      return true;
  }
}
inline bool opencl_is_unary_op_infix(UnaryOpType op) {
  switch (op) {
    case UnaryOpType::neg:
    case UnaryOpType::bit_not:
    case UnaryOpType::logic_not:
      return true;
    default:
      return false;
  }
}

// TODO: move this to lang_util.h:
inline std::string unary_op_type_symbol(UnaryOpType op) {
  switch (op) {
    case UnaryOpType::neg:
      return "-";
    case UnaryOpType::bit_not:
      return "~";
    case UnaryOpType::logic_not:
      return "!";
    default:
      return unary_op_type_name(op);
  }
}

// TODO: move this to lang_util.h:
inline DataType real_to_integral(DataType dt) {
  TI_ASSERT(is_real(dt));
  if (dt->is_primitive(PrimitiveTypeID::f16))
    return PrimitiveType::i16;
  else if (dt->is_primitive(PrimitiveTypeID::f32))
    return PrimitiveType::i32;
  else if (dt->is_primitive(PrimitiveTypeID::f64))
    return PrimitiveType::i64;
  else
    return PrimitiveType::unknown;
}

}  // namespace opencl
TLANG_NAMESPACE_END
