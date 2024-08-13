// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./SetBlockIdsRangePass.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/Dialect/HAL/Utils/LLVMLinkerUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/cuda_executable_def_builder.h"
#include "iree_cuda/libdevice_embedded.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {
struct CUDAOptions {
  bool dumpPtx = false;
  std::string clTargetChip = "sm_60";
  std::string clTargetFeature = "+ptx76";
  bool clUsePtxas = false;
  std::string clUsePtxasFrom;
  std::string clUsePtxasParams;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("CUDA HAL Target");
    binder.opt<bool>("iree-hal-cuda-dump-ptx", dumpPtx, llvm::cl::cat(category),
                     llvm::cl::desc("Dump ptx to the debug stream."));

    binder.opt<std::string>("iree-hal-cuda-llvm-target-arch", clTargetChip,
                            llvm::cl::cat(category),
                            llvm::cl::desc("LLVM target chip."));

    binder.opt<std::string>("iree-hal-cuda-llvm-target-feature",
                            clTargetFeature, llvm::cl::cat(category),
                            llvm::cl::desc("Use to set PTX version."));

    binder.opt<bool>(
        "iree-hal-cuda-use-ptxas", clUsePtxas, llvm::cl::cat(category),
        llvm::cl::desc("It uses the ptxas compiler that is on the environment, "
                       "compiles the "
                       "generated PTX code with it, puts the cubin binary "
                       "generated by ptxas "
                       "into the executable. "
                       "'--iree-hal-cuda-llvm-target-arch' is used as "
                       "the target GPU. If passing additional parameters to "
                       "ptxas is desired, "
                       "the parameters flag can be used "
                       "(e.g.'--iree-hal-cuda-use-ptxas-params=-v')."));

    binder.opt<std::string>(
        "iree-hal-cuda-use-ptxas-from", clUsePtxasFrom, llvm::cl::cat(category),
        llvm::cl::desc(
            "It uses the provided ptxas compiler, compiles the generated PTX "
            "code with it, puts the cubin binary generated by ptxas into the "
            "executable. '--iree-hal-cuda-llvm-target-arch' is used as the "
            "target GPU. If passing additional parameters to ptxas is desired, "
            "the "
            "parameters flag can be used "
            "(e.g.'--iree-hal-cuda-use-ptxas-params=-v')."));

    binder.opt<std::string>(
        "iree-hal-cuda-use-ptxas-params", clUsePtxasParams,
        llvm::cl::cat(category),
        llvm::cl::desc("Passes the given additional parameters to ptxas."));
  }

  LogicalResult verify(mlir::Builder &builder) const {
    if (GPU::normalizeCUDATarget(clTargetChip).empty()) {
      return emitError(builder.getUnknownLoc(), "Unknown CUDA target '")
             << clTargetChip << "'";
    }
    return success();
  }
};
} // namespace

static constexpr char kPtxasCompilerName[] = "ptxas";

/// Attempts to find ptxas compiler
static FailureOr<std::string> findPtxasCompiler(const CUDAOptions &options,
                                                std::string *message) {
  std::string ptxasCompiler;
  if (!options.clUsePtxasFrom.empty())
    ptxasCompiler = options.clUsePtxasFrom;
  if (llvm::sys::fs::exists(ptxasCompiler))
    return ptxasCompiler;

  ptxasCompiler = findTool(kPtxasCompilerName);
  if (llvm::sys::fs::exists(ptxasCompiler))
    return ptxasCompiler;

  *message = std::string(
      "Could not find ptxas compiler. Try passing it explicitly with "
      "--iree-hal-cuda-use-ptxas-from=<path> flag");
  return failure();
}

/// Compiles the given generated PTX code with the given ptxas compiler.
static FailureOr<std::string> compileWithPtxas(StringRef ptxasCompiler,
                                               StringRef smCapability,
                                               StringRef ptxasParams,
                                               StringRef ptxSource,
                                               std::string *message) {
  // Step 1. Create temporary files: ptx source file, log file and cubin file
  llvm::SmallString<64> ptxSourceFile, stdinFile, stdoutFile, stderrFile;
  llvm::sys::fs::createTemporaryFile("iree-ptx", "", ptxSourceFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stdin", "", stdinFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stdout", "", stdoutFile);
  llvm::sys::fs::createTemporaryFile("ptxas-stderr", "", stderrFile);
  std::string cubinFile = std::string(ptxSourceFile) + ".cubin";
  llvm::FileRemover stdinRemover(stdinFile.c_str());
  llvm::FileRemover stdoutRemover(stdoutFile.c_str());
  llvm::FileRemover stderrRemover(stderrFile.c_str());
  llvm::FileRemover binRemover(cubinFile.c_str());
  llvm::FileRemover srcRemover(ptxSourceFile.c_str());

  // Step 2. Write the generated PTX into a file, so we can pass it to ptxas
  // compiler
  std::error_code ec;
  llvm::raw_fd_ostream fPtxSource(ptxSourceFile, ec);
  fPtxSource << ptxSource;
  fPtxSource.close();
  if (fPtxSource.has_error()) {
    *message = std::string(
        "Could not write the generated ptx into a temporary file\n");
    return failure();
  }

  // Step 3. Build the ptxas command line
  std::vector<StringRef> ArgVector{
      StringRef(kPtxasCompilerName), StringRef("-arch"), smCapability,
      StringRef(ptxSourceFile),      StringRef("-o"),    StringRef(cubinFile)};
#ifdef _WIN32
  auto Tokenize = llvm::cl::TokenizeWindowsCommandLine;
#else
  auto Tokenize = llvm::cl::TokenizeGNUCommandLine;
#endif // _WIN32
  llvm::BumpPtrAllocator scratchAllocator;
  llvm::StringSaver stringSaver(scratchAllocator);
  SmallVector<const char *> rawArgs;
  Tokenize(ptxasParams, stringSaver, rawArgs, /*MarkEOLs=*/false);
  for (auto rawArg : rawArgs)
    ArgVector.push_back(StringRef(rawArg));

  std::optional<StringRef> redirects[] = {
      stdinFile.str(),
      stdoutFile.str(),
      stderrFile.str(),
  };

  // Step 4. Invoke ptxas
  if (llvm::sys::ExecuteAndWait(unescapeCommandLineComponent(ptxasCompiler),
                                llvm::ArrayRef<llvm::StringRef>(ArgVector),
                                /*Env=*/std::nullopt,
                                /*Redirects=*/redirects,
                                /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                /*ErrMsg=*/message)) {
    if (message->empty()) {
      *message = std::string("Invoking ptxas is failed, see the file: ") +
                 stderrFile.str().str() + std::string("\n");
    }
    stderrRemover.releaseFile();
    return failure();
  }

  // Step 5. The output of ptxas if verbose flag is set. This is useful
  // because it shows local memory usage, register usage, and etc.
  if (ptxasParams.find("-v") != StringRef::npos) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeFlog =
        llvm::MemoryBuffer::getFile(stderrFile);
    if (maybeFlog) {
      llvm::WithColor::note() << maybeFlog->get()->getBuffer().str();
    }
  }

  // Step 6. Read the cubin file, and return. It will eventually be written
  // into executable.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeFcubin =
      llvm::MemoryBuffer::getFile(cubinFile);
  if (!maybeFcubin) {
    *message = std::string("Could not read cubin file \n");
    return failure();
  }

  return std::string(maybeFcubin->get()->getBuffer());
}

// Attempt compiling the PtxImage with ptxas compiler. If the compilation fails
// for some reason return and pack the generated PtxImage code in the
// executable, let the runtime compile.
static std::string produceGpuImage(const CUDAOptions &options,
                                   StringRef targetArch,
                                   std::string &ptxImage) {
  if (!options.clUsePtxas)
    return ptxImage;

  std::string message;
  FailureOr<std::string> ptxasCompiler = findPtxasCompiler(options, &message);

  if (succeeded(ptxasCompiler)) {
    FailureOr<std::string> maybeCubinImage =
        compileWithPtxas(ptxasCompiler.value(), targetArch,
                         options.clUsePtxasParams, ptxImage, &message);
    if (succeeded(maybeCubinImage))
      return maybeCubinImage.value();
  }

  llvm::WithColor::warning()
      << "Compilation with `ptxas` failed, the generated ptx will be "
         "packaged into the executable and compiled at runtime. \n Error : "
      << message << " \n";

  return ptxImage;
}

static void dumpLLVMModuleToPath(StringRef path, StringRef baseName,
                                 StringRef suffix, StringRef extPrefix,
                                 llvm::Module &module) {
  // Dump disassembly to path.
  llvm::SmallVector<char> textData;
  llvm::raw_svector_ostream textOstream(textData);

  module.print(textOstream, nullptr);
  std::string textExtension = extPrefix.str() + ".ll";
  dumpDataToPath(path, baseName, suffix, textExtension,
                 StringRef(textData.data(), textData.size()));

  // Dump bitcode to path.
  llvm::SmallVector<char> binaryData;
  llvm::raw_svector_ostream binaryOstream(binaryData);
  // Write the specified module to the specified output stream.
  llvm::WriteBitcodeToFile(module, binaryOstream);
  std::string binaryExtension = extPrefix.str() + ".bc";
  dumpDataToPath(path, baseName, suffix, binaryExtension,
                 StringRef(binaryData.data(), binaryData.size()));
}

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

/// Resolve __nv function by linking libdevice module.
/// |objectAttrs| may optionally specify additional bitcode files to link into
/// the generated code.
static LogicalResult linkObjects(Location loc, llvm::Module &module,
                                 llvm::TargetMachine &targetMachine,
                                 ArrayAttr objectAttrs) {
  // Ensure consistent target information.
  const llvm::Triple &targetTriple = targetMachine.getTargetTriple();
  module.setDataLayout(targetMachine.createDataLayout());
  module.setTargetTriple(targetTriple.str());

  auto specializationCallback = [&](llvm::Module &userModule) {
    // TODO(thomasraoux): inject __nvvm_reflect-style functions/globals for
    // bitcode specialization based on the targetMachine and configuration.
    // These could use any information we have on the IREE side as well as the
    // TargetMachine instead of just what __nvvm_reflect supports (arch/etc).
  };

  // Link user modules and libdevice (if required).
  // Note that linking order matters:
  llvm::Linker linker(module);
  if (failed(linkCmdlineBitcodeFiles(loc, linker, llvm::Linker::OverrideFromSrc,
                                     targetMachine, module.getContext()))) {
    return failure();
  }

  unsigned linkerFlags =
      llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc;
  if (failed(linkBitcodeObjects(loc, linker, linkerFlags, targetMachine,
                                objectAttrs, module.getContext(),
                                specializationCallback))) {
    return mlir::emitError(loc)
           << "failed linking in user objects for target triple '"
           << targetTriple.str() << "'";
  }

  if (anyRequiredSymbols(module, "__nv_")) {
    llvm::MemoryBufferRef bitcodeBufferRef(
        llvm::StringRef(libdevice_embedded_create()->data,
                        libdevice_embedded_create()->size),
        "libdevice.xx.bc");
    if (failed(linkBitcodeModule(
            loc, linker, linkerFlags, targetMachine, "libdevice.xx.bc",
            llvm::parseBitcodeFile(bitcodeBufferRef, module.getContext())))) {
      return mlir::emitError(loc) << "failed linking in embedded libdevice "
                                     "bitcode for target triple '"
                                  << targetTriple.str() << "'";
    }
  }

  return success();
}

/// Performs optimizations on |module| (including LTO-style whole-program ones).
static void optimizeModule(llvm::Module &module,
                           llvm::TargetMachine &targetMachine,
                           const std::array<int32_t, 3> &maxWorkgroupSize) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  fam.registerPass([&] { return targetMachine.getTargetIRAnalysis(); });

  llvm::PipelineTuningOptions pto;
  pto.SLPVectorization = false;

  llvm::PassInstrumentationCallbacks pic;

  llvm::StandardInstrumentations si(module.getContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(&targetMachine, pto, std::nullopt, &pic);
  llvm::ModulePassManager mpm;
  StringRef nnvmReflectPassName = "nvvm-reflect";
  if (pb.parsePassPipeline(mpm, nnvmReflectPassName)) {
    llvm::errs() << "Could not parse -" << nnvmReflectPassName << "\n";
  }
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::OptimizationLevel ol = llvm::OptimizationLevel::O2;

  mpm.addPass(llvm::VerifierPass());
  llvm::FunctionPassManager fpm;
  fpm.addPass(llvm::SetBlockIdsRangePass(maxWorkgroupSize));
  mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
  mpm.addPass(pb.buildPerModuleDefaultPipeline(ol));
  mpm.addPass(llvm::VerifierPass());

  mpm.run(module, mam);
}

class CUDATargetDevice final : public TargetDevice {
public:
  CUDATargetDevice(const CUDAOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // TODO: device configuration attrs.
    auto configAttr = b.getDictionaryAttr(configItems);

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("cuda")->getDefaultExecutableTargets(
        context, "cuda", configAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("cuda"),
                                            configAttr, executableTargetAttrs);
  }

private:
  const CUDAOptions &options;
};

class CUDATargetBackend final : public TargetBackend {
public:
  CUDATargetBackend(const CUDAOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "cuda"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(b.getStringAttr(name), value);
    };

    if (failed(options.verify(b)))
      return nullptr;

    if (auto target = GPU::getCUDATargetDetails(
            options.clTargetChip, options.clTargetFeature, context))
      addConfig("iree.gpu.target", target);

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("cuda"), b.getStringAttr("cuda-nvptx-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // TODO: Derive the use of TransformDialect from inner
    // `LLVMGPULowerExecutableTargetPass`.
    registry.insert<gpu::GPUDialect, nvgpu::NVGPUDialect,
                    IREE::Codegen::IREECodegenDialect,
                    transform::TransformDialect, IREE::GPU::IREEGPUDialect>();
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
  }

  void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) override {
    buildLLVMGPUCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                    OpPassManager &passManager) override {
    buildLLVMGPUCodegenPassPipeline(passManager, false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    auto targetAttr = variantOp.getTargetAttr();
    StringRef targetArch = options.clTargetChip;
    StringRef targetFeatures = options.clTargetFeature;
    if (auto attr = getGPUTargetAttr(targetAttr)) {
      targetArch = attr.getArch();
      targetFeatures = attr.getFeatures();
    }

    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // TODO(thomasraoux): property handle export ordinals; this code is assuming
    // that ordinals are dense starting at 0 but that is not required.

    // Collect all the entry point parameters.
    SmallVector<std::array<int32_t, 3>> workgroupSizes;
    SmallVector<uint32_t> workgroupLocalMemories;
    for (auto exportOp : variantOp.getExportOps()) {
      std::array<int32_t, 3> workgroupSize;
      if (std::optional<ArrayAttr> workgroupSizeAttr =
              exportOp.getWorkgroupSize()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.value())) {
          workgroupSize[it.index()] =
              llvm::cast<IntegerAttr>(it.value()).getInt();
        }
      } else {
        workgroupSize = {1, 1, 1};
      }
      workgroupSizes.push_back(workgroupSize);
      uint32_t workgroupLocalMemory = 0;
      if (auto workgroupLocalMemoryAttr = exportOp.getWorkgroupLocalMemory()) {
        workgroupLocalMemory = workgroupLocalMemoryAttr->getSExtValue();
      }
      workgroupLocalMemories.push_back(workgroupLocalMemory);
    }

    FlatbufferBuilder builder;
    iree_hal_cuda_ExecutableDef_start_as_root(builder);

    // Attach embedded source file contents.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    SmallVector<std::string> entryPointNames;
    std::string ptxImage;
    SmallVector<iree_hal_debug_FileLineLocDef_ref_t> sourceLocationRefs;
    if (variantOp.isExternal()) {
      if (!variantOp.getObjects().has_value()) {
        return variantOp.emitOpError()
               << "no objects defined for external variant";
      } else if (variantOp.getObjects()->getValue().size() != 1) {
        // For now we assume there will be exactly one object file.
        // In the future we will want to perform a linking step here and ideally
        // support _also_ linking in the codegen results.
        return variantOp.emitOpError() << "only one object reference is "
                                          "supported for external variants";
      }

      // Take exported names verbatim. The user must have already sanitized
      // these to match the names in their kernels. We don't support any kind of
      // mangling and if the user was silly enough to rely on nvcc C++ mangling
      // they'll have to figure that out.
      for (auto exportOp : variantOp.getExportOps()) {
        entryPointNames.emplace_back(exportOp.getSymName());
      }

      auto objectAttr = llvm::cast<IREE::HAL::ExecutableObjectAttr>(
          variantOp.getObjects()->getValue().front());
      if (auto data = objectAttr.loadData()) {
        ptxImage = data.value();
      } else {
        return variantOp.emitOpError()
               << "object file could not be loaded: " << objectAttr;
      }
    } else {
      ModuleOp innerModuleOp = variantOp.getInnerModule();

      auto llvmModule =
          mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
      if (!llvmModule) {
        return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                        "dialect to the native llvm::Module";
      }

      for (auto [exportOp, workgroupSize] :
           llvm::zip_equal(variantOp.getExportOps(), workgroupSizes)) {
        auto *llvmFunc = llvmModule->getFunction(exportOp.getName());
        if (llvmFunc->isDeclaration())
          continue;

        // setName will make sure the function name is unique.
        llvmFunc->setName(sanitizeSymbolName(exportOp.getName()));
        entryPointNames.emplace_back(llvmFunc->getName());

        auto *annotations =
            llvmModule->getOrInsertNamedMetadata("nvvm.annotations");
        auto setMetadataValueI32 = [&](StringRef name, int value) {
          llvm::Metadata *llvmMetadata[] = {
              llvm::ValueAsMetadata::get(llvmFunc),
              llvm::MDString::get(llvmModule->getContext(), name),
              llvm::ValueAsMetadata::get(llvm::ConstantInt::get(
                  llvm::Type::getInt32Ty(llvmModule->getContext()), value))};
          annotations->addOperand(
              llvm::MDNode::get(llvmModule->getContext(), llvmMetadata));
        };
        // Mark the entry point as a kernel.
        setMetadataValueI32("kernel", 1);
        // Set the maximum number of threads in the thread block (CTA).
        setMetadataValueI32("maxntidx", workgroupSize[0]);
        setMetadataValueI32("maxntidy", workgroupSize[1]);
        setMetadataValueI32("maxntidz", workgroupSize[2]);

        // Optional source location information for debugging/profiling.
        if (serOptions.debugLevel >= 1) {
          if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
            auto filenameRef = builder.createString(loc->getFilename());
            sourceLocationRefs.push_back(iree_hal_debug_FileLineLocDef_create(
                builder, filenameRef, loc->getLine()));
          }
        }
      }

      std::unique_ptr<llvm::TargetMachine> targetMachine;
      {
        llvm::Triple triple("nvptx64-nvidia-cuda");
        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget("", triple, error);
        if (target == nullptr) {
          return variantOp.emitError() << "cannot initialize target triple";
        }
        targetMachine.reset(target->createTargetMachine(
            triple.str(), targetArch, targetFeatures, {}, {}));
        if (targetMachine == nullptr) {
          return variantOp.emitError() << "cannot initialize target machine";
        }
      }

      // Dump just the codegen bitcode before linking and optimization.
      if (!serOptions.dumpIntermediatesPath.empty()) {
        dumpLLVMModuleToPath(serOptions.dumpIntermediatesPath,
                             serOptions.dumpBaseName, variantOp.getName(),
                             ".codegen", *llvmModule);
      }

      // Link user and device bitcode alongside the generated module.
      if (failed(linkObjects(variantOp.getLoc(), *llvmModule, *targetMachine,
                             variantOp.getObjectsAttr()))) {
        return failure();
      }

      // Dump all linked bitcode prior to optimization.
      if (!serOptions.dumpIntermediatesPath.empty()) {
        dumpLLVMModuleToPath(serOptions.dumpIntermediatesPath,
                             serOptions.dumpBaseName, variantOp.getName(),
                             ".linked", *llvmModule);
      }

      std::array<int32_t, 3> maxWorkgroupSize = {1, 1, 1};
      for (int64_t i = 0, e = workgroupSizes.size(); i < e; i++) {
        for (int64_t j = 0; j < maxWorkgroupSize.size(); j++) {
          maxWorkgroupSize[j] =
              std::max(maxWorkgroupSize[j], workgroupSizes[i][j]);
        }
      }
      // Run LTO-style full optimization on the linked modules.
      optimizeModule(*llvmModule, *targetMachine, maxWorkgroupSize);

      // Dump bitcode post-linking and optimization.
      if (!serOptions.dumpIntermediatesPath.empty()) {
        dumpLLVMModuleToPath(serOptions.dumpIntermediatesPath,
                             serOptions.dumpBaseName, variantOp.getName(),
                             ".optimized", *llvmModule);
      }

      // Serialize CUDA kernel into the binary that we will embed in the
      // final FlatBuffer.
      ptxImage = translateModuleToISA(*llvmModule, *targetMachine);
    }

    if (options.dumpPtx) {
      llvm::dbgs() << ptxImage;
    }
    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".ptx", ptxImage);
    }

    std::string gpuImage = produceGpuImage(options, targetArch, ptxImage);
    auto gpuImageRef =
        flatbuffers_string_create(builder, gpuImage.c_str(), gpuImage.size());
    iree_hal_cuda_BlockSize_vec_start(builder);
    for (const auto &workgroupSize : workgroupSizes) {
      iree_hal_cuda_BlockSize_vec_push_create(
          builder, workgroupSize[0], workgroupSize[1], workgroupSize[2]);
    }
    auto blockSizesRef = iree_hal_cuda_BlockSize_vec_end(builder);
    auto workgroupLocalMemoriesRef =
        builder.createInt32Vec(workgroupLocalMemories);
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_hal_cuda_ExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_hal_cuda_ExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_hal_cuda_ExecutableDef_shared_memory_size_add(
        builder, workgroupLocalMemoriesRef);
    iree_hal_cuda_ExecutableDef_ptx_image_add(builder, gpuImageRef);
    if (!sourceLocationRefs.empty()) {
      auto sourceLocationsRef =
          builder.createOffsetVecDestructive(sourceLocationRefs);
      iree_hal_cuda_ExecutableDef_source_locations_add(builder,
                                                       sourceLocationsRef);
    }
    iree_hal_cuda_ExecutableDef_source_files_add(builder, sourceFilesRef);
    iree_hal_cuda_ExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

private:
  const CUDAOptions &options;
};

namespace {
struct CUDASession
    : public PluginSession<CUDASession, CUDAOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    // #hal.device.target<"cuda", ...
    targets.add("cuda",
                [&]() { return std::make_shared<CUDATargetDevice>(options); });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    // #hal.executable.target<"cuda", ...
    targets.add("cuda", [&]() {
      LLVMInitializeNVPTXTarget();
      LLVMInitializeNVPTXTargetMC();
      LLVMInitializeNVPTXTargetInfo();
      LLVMInitializeNVPTXAsmPrinter();
      return std::make_shared<CUDATargetBackend>(options);
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_cuda(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::CUDASession>(
      "hal_target_cuda");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::IREE::HAL::CUDAOptions);
