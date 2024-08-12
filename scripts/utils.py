
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import simpledialog
import webbrowser
import os
import subprocess
import zipfile
import sys

class OptionDialog(simpledialog.Dialog):
    def __init__(self, parent, title, message, options):
        self.message = message
        self.options = options
        self.selection = None
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text=self.message).pack(pady=10)
        
        for option in self.options:
            tk.Button(master, text=option, command=lambda opt=option: self.set_and_exit(opt)).pack(pady=5)


    def set_and_exit(self, option):
        self.selection = option
        self.destroy()

    def buttonbox(self):
        # Override the method to do nothing, thus removing "Ok" and "Cancel" buttons.
        pass

class Setup():
    def __init__(self, root, logfile):
        self.root = root
        self.logFile = logfile

    def showMessage(self, message):
        messagebox.showinfo("Info", message)

    def askOptions(self, message, options):
        d = OptionDialog(self.root, "Query", message, options)
        return d.selection

    def askYesOrNo(self, message):
        return messagebox.askyesno("Query", message, parent=self.root)

    def askForPath(self, customMessage):
        return filedialog.askdirectory(title=customMessage, parent=self.root)

    def open_url(self, url):
        webbrowser.open(url)

    def remindManualDependencies(self):
        # Create the pop-up window
        window = tk.Toplevel(self.root)
        window.title("Dependencies Installation")

        instructions = [
            "Before proceeding make sure to have installed the following dependencies:",
            ("- CUDA 11.7", "https://developer.nvidia.com/cuda-toolkit-archive"),
            ("- CUDA 8.0", "https://developer.nvidia.com/cuda-80-ga2-download-archive"),
            ("- Clang 12", "https://releases.llvm.org/download.html"),
            ("- OptiX 8.0", "https://developer.nvidia.com/designworks/optix/download"),
            ("- LibTorch", "https://pytorch.org/get-started/locally/"),
            "For LibTorch Select Windows for the OS, Libtorch for the Package and CUDA 11.7 for the Compute Platform. Then download the Release and Debug Version and extract wherever you want.",
            "The folders you will be prompted to select later are the \"libtorch\" folder inside the extracted folder for the Release and Debug Binaries"
        ]

        for instruction in instructions:
            if isinstance(instruction, tuple):
                link = tk.Label(window, text=instruction[0], fg="blue", cursor="hand2")
                link.pack(anchor="w", padx=20)
                link.bind("<Button-1>", lambda e, url=instruction[1]: self.open_url(url))
            else:
                label = tk.Label(window, text=instruction)
                label.pack(anchor="w", padx=20)

        btn = tk.Button(window, text="Continue", command=window.destroy)
        btn.pack(pady=20)
        window.wait_window()

    def stepDoneBefore(self, stepName):
        # Check if stepName exists in the log file
        if os.path.exists(self.logFile):
            with open(self.logFile, "r") as log:
                for line in log:
                    if stepName in line:
                        return True
        return False

    def readValueFromLog(self, logKey):
        with open(self.logFile, 'r') as file:
            for line in file:
                if logKey in line:
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        return parts[1]
        return None

    def markStepDone(self, stepName):
        # Mark the step as done by appending to the log file
        with open(self.logFile, "a") as log:
            log.write(f"{stepName}\n")

    def checkAndLogStep(self, stepKey, logKey, defaultPath, promptMessage):
        stepPath = None
        
        if not self.stepDoneBefore(stepKey):
            stepPath = self.readValueFromLog(logKey)
            if not stepPath:
                stepPath = defaultPath

            if not os.path.isdir(stepPath):
                stepPath = self.askForPath(promptMessage)

            print(f"{logKey} located in {stepPath}")
            
            with open(self.logFile, "a") as log:
                log.write(f"{logKey}={stepPath}\n")
            
            self.markStepDone(stepKey)
        else:
            stepPath = self.readValueFromLog(logKey)
            print(f"{logKey} located in {stepPath}")  # Assuming echoGreen prints in green. You can replace this with a function to print in green.

        return stepPath

    def echoGreen(self, message):
        GREEN = '\033[92m'
        END = '\033[0m'
        print(f"{GREEN}{message}{END}")

    def echoRed(self, message):
        RED = '\033[91m'  # Red color ANSI escape code
        END = '\033[0m'   # Reset to default color ANSI escape code
        print(f"{RED}{message}{END}")

    def configureVcpkg(self, ):
        vcpkg_dir = ""
        if not self.stepDoneBefore("vcpkgInstallation"):
            answer = self.askYesOrNo("Do you want to use an existing vcpkg installation? (Yes for existing, No for new)")

            if answer:
                vcpkg_dir = self.askForPath("Choose the existing vcpkg directory")
                print(f"User selected existing vcpkg directory: {vcpkg_dir}")
            else:
                if os.path.isdir("./ext/vcpkg"):
                    print("Existing vcpkg directory found. Updating...")
                    try:
                        subprocess.run(['git', 'pull', 'origin', 'master'], cwd='./ext/vcpkg', check=True)
                    except subprocess.CalledProcessError:
                        print("Failed to update vcpkg. Please resolve any issues manually.")
                        exit(1)
                    vcpkg_dir = "./ext/vcpkg"
                else:
                    print("Cloning vcpkg into ./ext/")
                    subprocess.run(['git', 'clone', 'https://github.com/Microsoft/vcpkg.git', './ext/vcpkg'])
                    vcpkg_dir = "./ext/vcpkg"

            vcpkg_dir = os.path.realpath(vcpkg_dir)

            print(f"vcpkg_dir is: {vcpkg_dir}")
            os.chdir(vcpkg_dir)

            # Note: Modify this line to execute the corresponding script for your platform
            print("running bootstrap")
            subprocess.run(['bootstrap-vcpkg.bat'])  # On Linux: ['bootstrap-vcpkg.sh']
            print("ending bootstrap")

            # Install dependencies
            commands = [
                ['vcpkg', 'install', 'boost-any', 'boost-uuid', '--triplet=x64-windows-static'],
                ['vcpkg', 'install', 'openimageio', '--triplet=x64-windows-static'],
                ['vcpkg', 'install', 'imgui[docking-experimental,opengl3-binding,glfw-binding,win32-binding]', '--triplet=x64-windows-static-md', '--recurse'],
                # ['vcpkg', 'install', 'spdlog', '--triplet=x64-windows-static-md'],
                #['vcpkg', 'install', 'yaml-cpp', '--triplet=x64-windows-static-md'],
                ['vcpkg', 'install', 'assimp', '--triplet=x64-windows-static-md'],
                ['vcpkg', 'install', 'glfw3', '--triplet=x64-windows-static-md'],
                ['vcpkg', 'install', 'implot', '--triplet=x64-windows-static-md']
            ]

            for command in commands:
                subprocess.run(command)

            print(f"vcpkg Directory set to {vcpkg_dir}")

            with open(self.logFile, 'a') as file:
                file.write(f"vcpkg_dir={vcpkg_dir}\n")

            self.markStepDone("vcpkgInstallation")

        else:
            vcpkg_dir = self.readValueFromLog("vcpkg_dir")
            self.echoGreen("Vcpkg and dependencies already installed")

        return vcpkg_dir
        
    def prepareMDL(self, extFolder, vcpkg_dir, clangPath):
        if not self.stepDoneBefore("mdlBuild"):
            # Print message
            print("Configuring MDL Cmake")

            # Create a build directory if it doesn't exist
            os.makedirs(os.path.join(extFolder, 'MDL-SDK', 'build'), exist_ok=True)

            # Navigate to the new build directory
            os.chdir(os.path.join(extFolder, 'MDL-SDK', 'build'))

            # Run cmake and specify the parent directory as the source directory
            cmake_cmd = [
                "cmake",
                f"-DCMAKE_TOOLCHAIN_FILE={os.path.join(vcpkg_dir, 'scripts', 'buildsystems', 'vcpkg.cmake')}",
                "-DMDL_BUILD_CORE_EXAMPLES=OFF",
                "-DMDL_BUILD_DOCUMENTATION=OFF",
                "-DMDL_BUILD_SDK_EXAMPLES=OFF",
                "-DMDL_ENABLE_CUDA_EXAMPLES=OFF",
                "-DMDL_ENABLE_D3D12_EXAMPLES=OFF",
                "-DMDL_ENABLE_OPENGL_EXAMPLES=OFF",
                "-DMDL_ENABLE_QT_EXAMPLES=OFF",
                "-DMDL_ENABLE_VULKAN_EXAMPLES=OFF",
                f"-Dclang_PATH={clangPath}",
                ".."
            ]
            subprocess.run(cmake_cmd)
            result = subprocess.run(cmake_cmd)
            if result.returncode != 0:
                self.echoRed("CMake configuration failed")
                sys.exit(1)

            print("Building MDL")
            result = subprocess.run(["cmake", "--build", ".", "--config", "Release"])
            if result.returncode != 0:
                self.echoRed("MDL build failed")
                sys.exit(1)

            result = subprocess.run(["cmake", "--build", ".", "--config", "Release", "--target", "INSTALL"])
            if result.returncode != 0:
                self.echoRed("MDL install failed")
                sys.exit(1)

            self.markStepDone("mdlBuild")

        else:
            self.echoGreen("MDL-SDK library already built")

    def updateTCNNFmtSubmodule(self, extFolder):
        tcnn_path = os.path.join(extFolder, 'tcnn')
        fmt_path = os.path.join(tcnn_path, 'dependencies', 'fmt')

        # Check if the fmt submodule directory exists
        if not os.path.isdir(fmt_path):
            print("fmt submodule directory not found.")
            return

        # Navigate to the fmt submodule and pull the latest changes from origin/master
        print("Updating fmt submodule to latest commit on origin/master...")
        update_cmds = [
            ["git", "fetch", "origin"],
            ["git", "checkout", "master"],
            ["git", "pull", "origin", "master"]
        ]

        for cmd in update_cmds:
            result = subprocess.run(cmd, cwd=fmt_path)
            if result.returncode != 0:
                print("Error updating fmt submodule. Command failed:", ' '.join(cmd))
                return

        # Update the main repository to point to the new commit of the submodule
        # print("Updating main repository to point to the updated fmt submodule...")
        # update_main_repo_cmd = ["git", "add", fmt_path]
        # result = subprocess.run(update_main_repo_cmd, cwd=tcnn_path)
        # if result.returncode != 0:
        #     print("Error updating main repository for fmt submodule.")
        #     return

        print("fmt submodule successfully updated.")
        
    def prepareTinyCudaNN(self, extFolder):
        tcnn_path = os.path.join(extFolder, 'tcnn')
        if not self.stepDoneBefore("tcnnBuild"):
            print("Configuring tiny-cuda-nn CMake")

            # Create a build directory if it doesn't exist
            os.makedirs(os.path.join(tcnn_path, 'build'), exist_ok=True)

            # Configure the project
            cmake_cmd = [
                "cmake",
                ".",
                "-B", "build",
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DTCNN_CUDA_ARCHITECTURES=75;86"
            ]
            result = subprocess.run(cmake_cmd, cwd=tcnn_path)
            if result.returncode != 0:
                self.echoRed("CMake configuration for tiny-cuda-nn failed")
                sys.exit(1)

            # Build the project
            print("Building tiny-cuda-nn")
            result = subprocess.run(["cmake", "--build", "build", "--config", "RelWithDebInfo", "-j"], cwd=tcnn_path)
            if result.returncode != 0:
                self.echoRed("Build of tiny-cuda-nn failed")
                sys.exit(1)

            self.markStepDone("tcnnBuild")

        else:
            self.echoGreen("tiny-cuda-nn already built")

    def prepareVortex(self, repoRoot, vcpkg_dir, torchPathDebug,torchPathRelease,cudaToolkitRoot,clang12Path,cuda8Path,optixPath):
        self.echoGreen("Configuring Vortex Cmake..")
        os.makedirs(os.path.join(repoRoot, 'build'), exist_ok=True)
        os.chdir(os.path.join(repoRoot, 'build'))
        # If you want to remove the cache and files, uncomment the next two lines
        # if os.path.exists('CMakeCache.txt'):
        #     os.remove('CMakeCache.txt')
        # if os.path.exists('CMakeFiles'):
        #     shutil.rmtree('CMakeFiles')
        cmake_cmd = [
                "cmake",
                f"-T cuda={cudaToolkitRoot}",
                f"-DCMAKE_TOOLCHAIN_FILE={os.path.join(vcpkg_dir, 'scripts', 'buildsystems', 'vcpkg.cmake')}",
                f"-DTORCH_INSTALL_PREFIX_DEBUG={torchPathDebug}",
                f"-DTORCH_INSTALL_PREFIX_RELEASE={torchPathRelease}",
                f"-DCUDAToolkit_ROOT={cudaToolkitRoot}",
                f"-DCLANG_12_PATH={clang12Path}",
                f"-DCUDA_8_PATH={cuda8Path}",
                f"-DOPTIX_PATH={optixPath}",
                ".."
            ]
        result = subprocess.run(cmake_cmd)
        if result.returncode != 0:
            self.echoRed("Vortex Build Failed")
            sys.exit(1)

        self.echoGreen("Cmake Configuration Completed!")

    def buildOrOpenVortex(self, repoRoot):
        message = "Set up completed! Do you want to build Vortex or open Visual Studio or Exit?"
        options = ["Build Vortex", "Open Visual Studio", "Exit"]

        choice = self.askOptions(message, options)
        
        if choice == "Build Vortex":
            self.echoGreen("Building Vortex...")
            subprocess.run(["cmake", "--build", ".", "--config", "Release"])
            self.echoGreen("Build complete, Launching Vortex.")
            os.chdir('./Vortex/src/Release/')
            os.startfile('Vortex.exe')  # This will launch the exe
        elif choice == "Open Visual Studio":
            self.echoGreen("Opening Visual Studio solution.")
            os.chdir(os.path.join(repoRoot, 'build'))
            os.startfile('Vortex.sln')
        elif choice == "Exit":
            self.echoGreen("Exiting.")
        else:
            self.echoGreen("Exiting.")

        input("Press any key to continue...")

    def unzipDemoScene(self, repoRoot):
        print("Unzipping Demo Scene")
        demoScene = repoRoot + "/assets/model/DemoSpaceship.zip"
        destination = repoRoot + "/assets/model/"
        with zipfile.ZipFile(demoScene, 'r') as zip_ref:
            zip_ref.extractall(destination)