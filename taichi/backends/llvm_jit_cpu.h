// A LLVM JIT compiler wrapper
#pragma once

// Based on
// https://llvm.org/docs/tutorial/BuildingAJIT3.html

// Note that
// https://llvm.org/docs/tutorial/BuildingAJIT2.html
// leads to a JIT that crashes all C++ exception after JIT session
// destruction...

#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include <memory>
#include "../tlang_util.h"
#include "jit_session.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name,
                           void *);
void global_optimize_module_cpu(std::unique_ptr<llvm::Module> &module);

class JITSessionCPU : public JITSession {
 private:
  ExecutionSession ES;
  std::map<VModuleKey, std::shared_ptr<SymbolResolver>> resolvers;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  LegacyRTDyldObjectLinkingLayer object_layer;
  LegacyIRCompileLayer<decltype(object_layer), SimpleCompiler> compile_layer;

  using OptimizeFunction = std::function<std::unique_ptr<llvm::Module>(
      std::unique_ptr<llvm::Module>)>;

  LegacyIRTransformLayer<decltype(compile_layer), OptimizeFunction>
      OptimizeLayer;

  std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
  LegacyCompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

 public:
  JITSessionCPU(JITTargetMachineBuilder JTMB, DataLayout DL)
      : TM(EngineBuilder().selectTarget()),
        DL(TM->createDataLayout()),
        object_layer(ES,
                     [this](VModuleKey K) {
                       return LegacyRTDyldObjectLinkingLayer::Resources{
                           std::make_shared<SectionMemoryManager>(),
                           resolvers[K]};
                     }),
        compile_layer(object_layer, SimpleCompiler(*TM)),
        OptimizeLayer(compile_layer,
                      [this](std::unique_ptr<llvm::Module> M) {
                        return optimize_module(std::move(M));
                      }),
        CompileCallbackManager(cantFail(
            orc::createLocalCompileCallbackManager(TM->getTargetTriple(),
                                                   ES,
                                                   0))),
        CODLayer(ES,
                 OptimizeLayer,
                 [&](orc::VModuleKey K) { return resolvers[K]; },
                 [&](orc::VModuleKey K, std::shared_ptr<SymbolResolver> R) {
                   resolvers[K] = std::move(R);
                 },
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *CompileCallbackManager,
                 orc::createLocalIndirectStubsManagerBuilder(
                     TM->getTargetTriple())) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  const DataLayout &get_data_layout() const override {
    return DL;
  }

  VModuleKey add_module(std::unique_ptr<llvm::Module> M) override {
    TI_ASSERT(M);
    global_optimize_module_cpu(M);
    // Create a new VModuleKey.
    VModuleKey K = ES.allocateVModule();

    // Build a resolver and associate it with the new key.
    resolvers[K] = createLegacyLookupResolver(
        ES,
        [this](const std::string &Name) -> JITSymbol {
          if (auto Sym = compile_layer.findSymbol(Name, false))
            return Sym;
          else if (auto Err = Sym.takeError())
            return std::move(Err);
          if (auto SymAddr =
                  RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return JITSymbol(SymAddr, JITSymbolFlags::Exported);
          return nullptr;
        },
        [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); });

    // Add the module to the JIT with the new key.
    cantFail(CODLayer.addModule(K, std::move(M)));
    return K;
  }

  JITSymbol lookup(const std::string Name) override {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CODLayer.findSymbol(MangledNameStream.str(), true);
  }

  void remove_module(VModuleKey K) override {
    cantFail(CODLayer.removeModule(K));
  }

  std::size_t get_type_size(llvm::Type *type) const override {
    return DL.getTypeAllocSize(type);
  }

 private:
  std::unique_ptr<llvm::Module> optimize_module(
      std::unique_ptr<llvm::Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    return M;
  }
};

inline std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch) {
  std::unique_ptr<JITTargetMachineBuilder> jtmb;
  if (arch_is_cpu(arch)) {
    auto JTMB = JITTargetMachineBuilder::detectHost();
    if (!JTMB)
      TI_ERROR("LLVM TargetMachineBuilder has failed.");
    jtmb = std::make_unique<JITTargetMachineBuilder>(std::move(*JTMB));
  } else {
    TI_ASSERT(arch == Arch::cuda);
    Triple triple("nvptx64", "nvidia", "cuda");
    jtmb = std::make_unique<JITTargetMachineBuilder>(triple);
    TI_WARN("Not actually supported");
  }

  auto DL = jtmb->getDefaultDataLayoutForTarget();
  if (!DL) {
    TI_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }

  return llvm::make_unique<JITSessionCPU>(std::move(*jtmb), std::move(*DL));
}

TLANG_NAMESPACE_END
