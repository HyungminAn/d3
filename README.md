# D3 dispersion correction on LAMMPS with CUDA

Only NVIDIA GPU supported.

This is for avoiding collision between openACC and pyTorch.

The parallelization used is the same as openACC version.

## Installation

### Compile CUDA D3 on LAMMPS
Requirements
- compiler supporting CUDA nvcc (g++ 12.1.1 tested)
- LAMMPS (`23Jun2022` tested)

My environment
- Module: compiler/2022.1.0 mpi/2021.6.0 mkl/2022.1.0 CUDA/12.1.0 (odin/loki server)

-----
1. Copy `pair_d3.cu` and `pair_d3.h` into the lammps/src directory (not available with CPU version D3 `pair_d3.cpp`)

2. Configure `CMakeLists.txt` in the lammps/cmake directory
  - Change: `${LAMMPS_SOURCE_DIR}/[^.]*.cpp` -> `${LAMMPS_SOURCE_DIR}/[^.]*.cpp  ${LAMMPS_SOURCE_DIR}/[^.]*.cu`
  - Add to the last line:
    ```
    find_package(CUDA)
    target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)
    ```

3. Enter command in the lammps directory
  ```
  mkdir build
  cd build

  cmake ../cmake -C ../cmake/presets/gcc.cmake \
  -D BUILD_MPI=no -D BUILD_OMP=no \
  -D CMAKE_CXX_FLAGS="-O3" \
  -D CMAKE_CUDA_FLAGS="-fmad=false -O3" \
  -D CMAKE_CUDA_ARCHITECTURES="86;80;70;61"

  make -j8
  ```

### Compile CUDA D3 with SevenNet on LAMMPS
Requirements
- libtorch (pre-cxx11 and cxx11 tested)
- compiler supporting CUDA nvcc (g++ 12.1.1 tested)
- LAMMPS (`23Jun2022` tested)

My environment
- Module: compiler/2022.1.0 mpi/2021.6.0 mkl/2022.1.0 CUDA/12.1.0 (odin/loki server)
- Conda: pub_sevenn (SevenNet uses libtorch of this env)

-----
1. Copy `pair_d3.cu` and `pair_d3.h` into the lammps/src directory (not available with CPU version D3 `pair_d3.cpp`)


2. Configure `CMakeLists.txt` in the lammps/cmake directory
  - Change: `set(CMAKE_CXX_STANDARD 11)` -> `set(CMAKE_CXX_STANDARD 14)`
  - Change: `${LAMMPS_SOURCE_DIR}/[^.]*.cpp` -> `${LAMMPS_SOURCE_DIR}/[^.]*.cpp  ${LAMMPS_SOURCE_DIR}/[^.]*.cu`
  - Add to the last line:
    ```
    find_package(CUDA)
    target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)

    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
    ```

3. Enter command in the lammps directory
  ```
  mkdir build
  cd build

  cmake ../cmake -C ../cmake/presets/gcc.cmake \
  -D BUILD_MPI=no -D BUILD_OMP=no \
  -D CMAKE_CXX_FLAGS="-O3" \
  -D CMAKE_CUDA_FLAGS="-fmad=false -O3" \
  -D CMAKE_CUDA_ARCHITECTURES="86;80;70;61" \
  -D CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")

  make -j8
  ```

### Notes
- `fmad=false` is essential to obtain precise figures. Be careful to ensure that the result value is correct.
- CMAKE_CUDA_ARCHITECUTRES lists
  - 61 -> Titan X, P6000
  - 70 -> v100
  - 80 -> a100
  - 86 -> 3090ti, a5000
- If there is a GPU on the node you are compiling, CMake will find it, so setting CMAKE_CUDA_ARCHITECTURES is unnecessary (maybe).
- If there is no GPU in the node compiling, CMake can cause errors.

## To do
- Implement without Unified Memory.
- Unfix the threadsPerBlock=128.
- Achieve more effective parallelism.

## Cautions
- It can be slower than the CPU with a small number of atoms.
- The CUDA math library differs from C, which can lead to numerical errors.