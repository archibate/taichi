#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#if defined(TI_WITH_CUDA)
#include <taichi/cuda_utils.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif
#include "cuda_context.h"
#include "llvm_jit_cpu.h"
#include <taichi/program.h>
#include <taichi/context.h>
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)

std::string cuda_mattrs() {
  return "+ptx50";
}

std::unique_ptr<CUDAContext> cuda_context;  // TODO:..

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  // Part of this function is borrowed from Halide::CodeGen_PTX_Dev.cpp
  using namespace llvm;

  llvm::Triple triple(module->getTargetTriple());

  // Allocate target machine

  std::string err_str;
  const llvm::Target *target =
      TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  bool fast_math = get_current_program().config.fast_math;

  TargetOptions options;
  options.PrintMachineCode = 0;
  if (fast_math) {
    options.AllowFPOpFusion = FPOpFusion::Fast;
    // See NVPTXISelLowering.cpp
    // Setting UnsafeFPMath true will result in approximations such as
    // sqrt.approx in PTX for both f32 and f64
    options.UnsafeFPMath = 1;
    options.NoInfsFPMath = 1;
    options.NoNaNsFPMath = 1;
  } else {
    options.AllowFPOpFusion = FPOpFusion::Strict;
    options.UnsafeFPMath = 0;
    options.NoInfsFPMath = 0;
    options.NoNaNsFPMath = 0;
  }
  options.HonorSignDependentRoundingFPMathOption = 0;
  options.NoZerosInBSS = 0;
  options.GuaranteedTailCallOpt = 0;
  options.StackAlignmentOverride = 0;

  std::unique_ptr<TargetMachine> target_machine(target->createTargetMachine(
      triple.str(), cuda_context->get_mcpu(), cuda_mattrs(), options,
      llvm::Reloc::PIC_, llvm::CodeModel::Small, CodeGenOpt::Aggressive));

  TI_ERROR_UNLESS(target_machine.get(), "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // Set up passes
  llvm::SmallString<8> outstr;
  raw_svector_ostream ostream(outstr);
  ostream.SetUnbuffered();

  legacy::FunctionPassManager function_pass_manager(module.get());
  legacy::PassManager module_pass_manager;

  module_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));
  function_pass_manager.add(createTargetTransformInfoWrapperPass(
      target_machine->getTargetIRAnalysis()));

  // NVidia's libdevice library uses a __nvvm_reflect to choose
  // how to handle denormalized numbers. (The pass replaces calls
  // to __nvvm_reflect with a constant via a map lookup. The inliner
  // pass then resolves these situations to fast code, often a single
  // instruction per decision point.)
  //
  // The default is (more) IEEE like handling. FTZ mode flushes them
  // to zero. (This may only apply to single-precision.)
  //
  // The libdevice documentation covers other options for math accuracy
  // such as replacing division with multiply by the reciprocal and
  // use of fused-multiply-add, but they do not seem to be controlled
  // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
  // passes.
  const auto kFTZDenorms = 1;

  // Insert a module flag for the FTZ handling.
  module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                        kFTZDenorms);

  if (kFTZDenorms) {
    for (llvm::Function &fn : *module) {
      fn.addFnAttr("nvptx-f32ftz", "true");
    }
  }

  PassManagerBuilder b;
  b.OptLevel = 3;
  b.Inliner = createFunctionInliningPass(b.OptLevel, 0, false);
  b.LoopVectorize = false;
  b.SLPVectorize = false;

  target_machine->adjustPassManager(b);

  b.populateFunctionPassManager(function_pass_manager);
  b.populateModulePassManager(module_pass_manager);

  // Override default to generate verbose assembly.
  target_machine->Options.MCOptions.AsmVerbose = true;

  // Output string stream

  // Ask the target to add backend passes as necessary.
  bool fail = target_machine->addPassesToEmitFile(
      module_pass_manager, ostream, nullptr, TargetMachine::CGFT_AssemblyFile,
      true);

  TI_ERROR_IF(fail, "Failed to set up passes to emit PTX source\n");

  // Run optimization passes
  function_pass_manager.doInitialization();
  for (llvm::Module::iterator i = module->begin(); i != module->end(); i++) {
    function_pass_manager.run(*i);
  }
  function_pass_manager.doFinalization();
  module_pass_manager.run(*module);

  std::string buffer(outstr.begin(), outstr.end());

  // Null-terminate the ptx source
  buffer.push_back(0);
  return buffer;
}

CUDAContext::CUDAContext() {
  // CUDA initialization
  dev_count = 0;
  check_cuda_error(cuInit(0));
  check_cuda_error(cuDeviceGetCount(&dev_count));
  check_cuda_error(cuDeviceGet(&device, 0));

  char name[128];
  check_cuda_error(cuDeviceGetName(name, 128, device));
  TI_TRACE("Using CUDA Device [id=0]: {}", name);

  int cc_major, cc_minor;
  check_cuda_error(cuDeviceGetAttribute(
      &cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  check_cuda_error(cuDeviceGetAttribute(
      &cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

  TI_TRACE("CUDA Device Compute Capability: {}.{}", cc_major, cc_minor);
  check_cuda_error(cuCtxCreate(&context, 0, device));
  check_cuda_error(cudaMalloc(&context_buffer, sizeof(Context)));

  mcpu = fmt::format("sm_{}{}", cc_major, cc_minor);
}

CUmodule CUDAContext::compile(const std::string &ptx) {
  // auto _ = cuda_context->get_guard();
  make_current();
  // Create module for object
  CUmodule cudaModule;
  TI_TRACE("PTX size: {:.2f}KB", ptx.size() / 1024.0);
  auto t = Time::get_time();
  TI_TRACE("Loading module...");
  auto _ = std::lock_guard<std::mutex>(cuda_context->lock);
  check_cuda_error(
      cuModuleLoadDataEx(&cudaModule, ptx.c_str(), 0, nullptr, nullptr));
  TI_TRACE("CUDA module load time : {}ms", (Time::get_time() - t) * 1000);
  cudaModules.push_back(cudaModule);
  return cudaModule;
}

CUfunction CUDAContext::get_function(CUmodule module,
                                     const std::string &func_name) {
  // auto _ = cuda_context->get_guard();
  make_current();
  CUfunction func;
  auto t = Time::get_time();
  check_cuda_error(cuModuleGetFunction(&func, module, func_name.c_str()));
  t = Time::get_time() - t;
  TI_TRACE("Kernel {} compilation time: {}ms", func_name, t * 1000);
  return func;
}

void CUDAContext::launch(CUfunction func,
                         const std::string &task_name,
                         ProfilerBase *profiler,
                         void *context_ptr,
                         unsigned gridDim,
                         unsigned blockDim) {
  // auto _ = cuda_context->get_guard();
  make_current();
  // Kernel parameters

  void *KernelParams[] = {context_ptr};

  if (profiler) {
    profiler->start(task_name);
  }
  // Kernel launch
  if (gridDim > 0) {
    std::lock_guard<std::mutex> _(lock);
    check_cuda_error(cuLaunchKernel(func, gridDim, 1, 1, blockDim, 1, 1, 0,
                                     nullptr, KernelParams, nullptr));
  }
  if (profiler) {
    profiler->stop();
  }

  if (get_current_program().config.debug) {
    check_cuda_error(cudaDeviceSynchronize());
    auto err = cudaGetLastError();
    if (err) {
      TI_ERROR("CUDA Kernel Launch Error: {}", cudaGetErrorString(err));
    }
  }
}

CUDAContext::~CUDAContext() {
  /*
  check_cuda_error(cuMemFree(context_buffer));
  for (auto cudaModule: cudaModules)
    check_cuda_error(cuModuleUnload(cudaModule));
  check_cuda_error(cuCtxDestroy(context));
  */
}

#else
std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module) {
  TI_NOT_IMPLEMENTED
}

int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name,
                           void *) {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
