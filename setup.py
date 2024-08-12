import tkinter as tk
import os
import subprocess
from scripts.utils import Setup

try:
    root = tk.Tk()
    root.withdraw()  # Always keep this root hidden and use it as a parent for everything else.

    

    # constants
    repoRoot = os.getcwd()
    extFolder = os.path.join(repoRoot, 'ext')
    logFile = os.path.join(repoRoot, 'installationLog.txt')

    setup = Setup(root, logFile)

    # the script needs to be runed from the repo dir
    if not os.path.exists("setup.py"):
        setup.showMessage("Please run this script from the repo root directory.")
        exit(1)

    # create the log file if it doesn't exist
    with open(logFile, 'a') as _:
        pass

    setup.remindManualDependencies()

    cudaToolkitRoot = setup.checkAndLogStep(
        "cudaToolkitChecked", "cudaToolkitRoot", 
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7", 
        "Please set CUDA 11.7 folder, usually is set to C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.7")

    cuda8Path = setup.checkAndLogStep(
        "cuda8Checked", "cuda8Path", 
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0", 
        "Please set CUDA 8.0 folder, usually is set to C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0")

    optixPath = setup.checkAndLogStep(
        "optixPathChecked", "optixPath", 
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0",
        "Please set Optix Path, usually is set to C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")

    clang12Path = setup.checkAndLogStep(
        "clang12Checked", "clang12Path", 
        "C:/Program Files (x86)/LLVM", 
        "Please set Clang 12 folder, usually is set to C:/Program Files (x86)/LLVM")
    clang12Path = os.path.join(clang12Path, "bin", "clang.exe")

    torchPathRelease = setup.checkAndLogStep(
        "libTorchReleaseChecked", "torchPathRelease", 
        "", 
        "Please set LibTorch Release folder. The folder path needs to point to the libtorch folder inside the extract folder from the downloaded compressed file.")

    torchPathDebug = setup.checkAndLogStep(
        "libTorchDebugChecked", "torchPathDebug", 
        "", 
        "Please set LibTorch Debug folder. The folder path needs to point to the libtorch folder inside the extract folder from the downloaded compressed file.")

    vcpkg_dir = setup.configureVcpkg()

    setup.prepareMDL(extFolder, vcpkg_dir, clang12Path)

    # setup.updateTCNNFmtSubmodule(extFolder)

    setup.unzipDemoScene(repoRoot)
    
    setup.prepareVortex(repoRoot, vcpkg_dir, torchPathDebug,torchPathRelease,cudaToolkitRoot,clang12Path,cuda8Path,optixPath)

    setup.buildOrOpenVortex(repoRoot)
except Exception as e:
    print(f"An error occurred: {e}")
    input("Press Enter to exit.")