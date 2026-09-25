# Building llmx

llmx builds from source with a C++17 compiler and CMake on Windows, Linux and macOS, or with `build.bat` alone on Windows.
It links no third-party libraries.
The CPU backend is always built; the optional Vulkan backend needs the Vulkan headers and the `glslc` shader compiler at build time, and the Vulkan loader and a driver at run time.
This page lists what each platform needs, the commands, where the binary lands and how to test it.
The [README](../README.md#build) keeps a short quick start.

## Requirements on every platform

- **An x86-64 CPU with AVX2, FMA and F16C.**
  The whole binary is compiled for them: `build.bat` and CMake with MSVC pass `/arch:AVX2`, and CMake with GCC or Clang passes `-mavx2 -mfma -mf16c` on x86-64.
  The runtime checks inside some kernels do not make that binary run on an older CPU.
  The CPU backend is written with x86 intrinsics, so ARM, Apple Silicon included, is not supported.
- **A C++17 compiler.**
  MSVC from Visual Studio 2022 or later on Windows, GCC or Clang on Linux, and Apple Clang on macOS.
  CI builds with MSVC, GCC and Apple Clang ([CI](CI.md)).
- **CMake 3.16 or later**, the minimum `CMakeLists.txt` sets, for every route except `build.bat`.
  CMake's Visual Studio generators need a newer one: 3.21 or later for Visual Studio 2022, and 4.2 or later for Visual Studio 2026.
  `ctest --test-dir`, used under [Tests](#after-building-the-tests), needs CTest 3.20 or later.
- **git**, for the build identifier that `llmx --version` prints.
  Without git, or in a tree that is not a git checkout, the build still works and reports `0.1.0+unknown`.
- **Python 3**, only to run the tests, which use its standard library alone.
  CI runs Python 3.12.
- **curl 8.4 or later** at run time, only for `llmx pull`, which runs curl as a child process to download models.
  On Windows that is the `curl.exe` in the Windows system directory, and elsewhere the first `curl` on the `PATH`.
  The build does not need it.

## Windows

Install Visual Studio 2022 or later, or the Build Tools for Visual Studio, with the **Desktop development with C++** workload.
Install git, and Python 3 to run the tests.

### `build.bat`: the CPU backend without CMake

From a Command Prompt in the repository root:

```bat
build.bat
llmx.exe --version
```

In PowerShell, type `.\build.bat` and `.\llmx.exe --version`.

`build.bat` asks `vswhere.exe`, which the Visual Studio Installer keeps in `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer`, for the newest installation that has the x64 C++ tools, and loads that installation's compiler environment through `vcvars64.bat`, so any Command Prompt or PowerShell window works.
If `vswhere.exe` is missing, or no installation has the C++ tools, it stops and says which.
It compiles `src\cli\main.cpp` in one `cl` call and writes `llmx.exe` to the repository root, with its version header in `build\plain-generated` and the object file `main.obj` beside `llmx.exe`; git ignores all three.
It builds the CPU backend only: the Vulkan backend and the native tests need CMake.
CI runs `build.bat` on its Windows runner too, requires the binary to report the same version as the CMake build, and runs the tests on it.

### CMake

```bat
cmake -S . -B build
cmake --build build --config Release --parallel
build\Release\llmx.exe --version
```

CMake picks the newest Visual Studio it knows as the generator.
That generator holds several configurations in one build directory, so `--config Release` selects the optimized build and the binary lands in `build\Release\`.
The cross-platform lines in the README also pass `-DCMAKE_BUILD_TYPE=Release`, which this generator ignores with a warning.
A standalone CMake works, and so does the one the workload's **C++ CMake tools for Windows** component installs, which a Developer Command Prompt puts on the `PATH`.

### The Vulkan backend on Windows

Install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) from LunarG.
It provides the Vulkan headers and `glslc`, and its installer sets the `VULKAN_SDK` environment variable, under which CMake looks for both (`%VULKAN_SDK%\Include` and `%VULKAN_SDK%\Bin`).
Open a new terminal after installing it, so the variable is set there.
Then configure with the option, build, and run the synthetic benchmark on the first device as a check that the device and the build work:

```bat
cmake -S . -B build -DLLMX_HAS_BACKEND_VULKAN=ON
cmake --build build --config Release --parallel
build\Release\llmx.exe bench --device vulkan:0
```

Without `vulkan/vulkan.h` or `glslc`, configuring stops with `LLMX_HAS_BACKEND_VULKAN needs vulkan/vulkan.h and glslc`.
The SDK is needed only to build.
At run time llmx loads `vulkan-1.dll`, which the GPU driver installs; on a machine without it, `--device vulkan:N` fails with a message and the CPU backend still runs.
The Windows results in [VULKAN](VULKAN.md) were built with SDK 1.4.357.

The option stays set in the build directory's cache.
Configure again with `-DLLMX_HAS_BACKEND_VULKAN=OFF` to go back, or keep the two builds side by side in separate directories, such as `build` and `build-vulkan`; git ignores both.

## Linux

### The CPU backend on Linux

On Debian or Ubuntu, install the compiler, CMake and git:

```sh
sudo apt-get install build-essential cmake git
```

Other distributions need the equivalent: `g++` or `clang++`, `make`, `cmake` and `git`.
Add `python3` to run the tests and `curl` for `llmx pull`.
Then, in the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
./build/llmx --version
```

The default generator, Unix Makefiles, holds one configuration per build directory, so `CMAKE_BUILD_TYPE` selects the optimized build and the binary lands at `build/llmx`.

### The Vulkan backend on Linux

The build needs the Vulkan headers and a `glslc` new enough for the shader extensions the kernels use.
Running on a device needs the Vulkan loader, `libvulkan.so.1`, and a driver for the card.

**Ubuntu 24.04.**
The distribution's own `glslc` is older than those shader extensions, so CI's Vulkan job takes the headers and the compiler from the LunarG repository, with these commands:

```sh
sudo wget -qO /etc/apt/trusted.gpg.d/lunarg.asc https://packages.lunarg.com/lunarg-signing-key-pub.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list https://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list
sudo apt-get update
sudo apt-get install -y --no-install-recommends shaderc vulkan-headers
```

LunarG's noble repository stopped at SDK 1.4.313, which is new enough; for a newer SDK, use the Linux archive described under **Elsewhere**.
CI builds there and runs the CTests, whose device tests skip, since hosted runners have no GPU.
To run on a card, the machine also needs the loader (package `libvulkan1`) and a driver, such as Mesa's `mesa-vulkan-drivers`, which includes RADV for AMD cards.

**Debian 13 (trixie).**
The distribution's packages are new enough.
These are the ones the Docker image below installs for the build and the device, and they bring the compiler, CMake, git, `glslc`, the headers, the loader and Mesa's drivers:

```sh
sudo apt-get install build-essential cmake git glslc libvulkan-dev mesa-vulkan-drivers
```

The image also installs `python3`, `curl`, `ca-certificates`, and `vulkan-tools` for `vulkaninfo`, which lists the devices the loader sees.

**Elsewhere,** CMake looks for `vulkan/vulkan.h` and `glslc` on the system paths and under `$VULKAN_SDK/include` and `$VULKAN_SDK/bin`, so any source of the headers and a recent `glslc` works, the Vulkan SDK's Linux archive included once `source setup-env.sh` has set `VULKAN_SDK` in that shell.

Then:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLMX_HAS_BACKEND_VULKAN=ON
cmake --build build --config Release --parallel
./build/llmx bench --device vulkan:0
```

**Access to the card.**
The driver opens the card's render node, `/dev/dri/renderD128` and up, which on Debian and Ubuntu belongs to the `render` group.
Add your user to that group and log in again:

```sh
sudo usermod -aG render "$USER"
```

### Docker, with nothing installed on the host

`docker/Dockerfile` builds a Debian trixie image with everything the Vulkan backend needs: the compiler, CMake, git, `glslc`, the headers, the loader, Mesa's drivers, Python and curl.
The host needs only Docker and the kernel driver for its cards, which reach the container through `/dev/dri`, so its own packages stay untouched.
From the repository root, build the image, then start a shell in it with the repository mounted at `/llmx` and the host's `render` group added so the container may open the render nodes:

```sh
docker build -t llmx-dev docker
docker run --rm -it --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD:/llmx" -w /llmx llmx-dev
```

Inside, the ordinary commands apply:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLMX_HAS_BACKEND_VULKAN=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/llmx bench --model <model.gguf> --device vulkan:0
```

The container runs as root, so the build directory it writes into the mounted tree belongs to root on the host.

## macOS

Only Intel Macs are supported, since the build needs AVX2, FMA and F16C; Apple Silicon is not supported.
Install the Xcode Command Line Tools, which bring Apple Clang, `make` and git, and install CMake, for example from Homebrew:

```sh
xcode-select --install
brew install cmake
```

Then, in the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
./build/llmx --version
```

This builds the CPU backend, the configuration CI builds and tests on its `macos-15-intel` runner.
The Vulkan backend is not supported on macOS: it opens the Vulkan loader only as `vulkan-1.dll` or `libvulkan.so.1`.

## Outputs and options

| Route | Binary |
|---|---|
| `build.bat` | `llmx.exe` in the repository root |
| CMake with a Visual Studio generator, the default on Windows | `build\Release\llmx.exe` |
| CMake with Makefiles or Ninja, Makefiles being the default on Linux and macOS | `build/llmx` |

Other generators that hold several configurations, such as Ninja Multi-Config or Xcode, also put the binary under `build/Release/`.

| CMake option | Default | Effect |
|---|---|---|
| `LLMX_HAS_BACKEND_VULKAN` | `OFF` | Builds the Vulkan backend into `llmx`, with every shader compiled by `glslc` at build time and embedded in the binary, so there are no shader files to ship |
| `BUILD_TESTING` | `ON` | Builds the native tests and tools and registers the tests with CTest; `OFF` builds only `llmx` and, in a Vulkan build, the backend library it links |

With `BUILD_TESTING` on, every build has `llmx` and the `llmx-*-test` programs behind the CTests; `llmx-prefill-placement-test` is built on Windows only.
These targets exist only with `LLMX_HAS_BACKEND_VULKAN=ON`:

- `llmx-vulkan`, the backend library the binary, tests and tools link, and `llmx-vulkan-shaders`, the step that compiles the shaders.
- `llmx-backend-vulkan-test`, the CTest `backend-vulkan`, and `llmx-vulkan-lifetime-test`, the CTests `vulkan-buffer` and `vulkan-lifetime`.
- The tools `llmx-split-check`, `llmx-multi-device-bench`, `llmx-moe-kernel-bench` and `llmx-vk-handoff`, which are not tests.

CTest runs the native tests in every configuration, and the Vulkan build adds the device tests above.

### The build identifier

`llmx --version` prints the release and the commit the binary was built from, for example `llmx 0.1.0+g0123456789ab`.

- `0.1.0` is the release: `project(llmx VERSION ...)` in `CMakeLists.txt`, and `LLMX_RELEASE_VERSION` in `src/config.hpp` for `build.bat`.
- `g` and at least 12 hex digits name the commit.
- `.dirty` follows when tracked files differ from that commit; untracked files do not count.
- `unknown` replaces the commit when git is missing or the source directory is not the top of a git checkout, as in a source archive.

Both routes refresh the identifier on every build, CMake without reconfiguring, and neither adds a timestamp or changes the release number.

## After building: the tests

The native tests run through CTest after a CMake build, and the Python suite drives the built binary.
[AGENTS](../AGENTS.md#tests) describes what each test covers, and [CI](CI.md) which of them each hosted job runs.

Windows, after `build.bat` (the suite's default binary is `llmx.exe` in the repository root, so no `--exe` is needed):

```bat
python tests/run_tests.py --no-perf-floor
```

Windows, after the CMake build:

```bat
ctest --test-dir build -C Release --output-on-failure
python tests/run_tests.py --exe build/Release/llmx.exe --no-perf-floor
```

Linux and macOS:

```sh
ctest --test-dir build -C Release --output-on-failure
python3 tests/run_tests.py --exe build/llmx --no-perf-floor
```

- `--no-perf-floor` reports the `perf` component's timings without enforcing its floors, which were set on the development workstation; CI passes it too.
- `--device vulkan:0` runs every command in the suite that takes `--device` on that device.
- `--only baseline` runs just the named components, comma separated for several.
- Real-model checks skip when their models are absent; `tools/fetch_test_models.py` downloads the pinned models, and `--require-baseline` makes a missing one fail the suite.
- In a Vulkan build without a usable device, CTest reports `backend-vulkan` and `vulkan-lifetime` as skipped.
