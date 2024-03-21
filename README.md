# D3 dispersion correction on LAMMPS with CUDA

Only NVIDIA GPU supported.

This is for avoiding collision between openACC and pyTorch.

The parallelization used is the same as openACC version.

## Usage

It requires only CUDA, not OpenMP or MPI.

Example LAMMPS input script:
```
pair_style d3 9000 1600 d3_damp_bj                     # Available d3_damp_zero d3_damp_bj
pair_coeff * * ./r0ab.csv ./d3_pars.csv pbe C H O      # Specify used elements
compute vp_d3 all pressure NULL virial pair/hybrid d3  # Necessary for pressure values
```

## Installation

### Compile CUDA D3 on LAMMPS
Requirements
- Compiler supporting CUDA nvcc (g++ 12.1.1 tested)
- LAMMPS (`23Jun2022` tested)

My environment
- Module: compiler/2022.1.0 mpi/2021.6.0 mkl/2022.1.0 CUDA/12.1.0 (odin/loki server)

-----
1. Copy `pair_d3.cu` and `pair_d3.h` into the lammps/src directory (not available with CPU version D3 `pair_d3.cpp`)

2. Configure `CMakeLists.txt` in the lammps/cmake directory
  - Change: `project(lammps CXX)` -> `project(lammps CXX CUDA)`
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
The description below is simply a combination of the above explanation with the compilation of SevenNet.

Requirements
- Libtorch (If you have installed **PyTorch**, then libtorch is already installed)
- Compiler supporting CUDA nvcc (g++ 12.1.1 tested)
- LAMMPS (`23Jun2022` tested)

My environment
- Module: compiler/2022.1.0 mpi/2021.6.0 mkl/2022.1.0 CUDA/12.1.0 (odin/loki server)
- Conda: pub_sevenn (SevenNet uses libtorch of this env)

-----
1. Copy `pair_d3.cu` and `pair_d3.h` into the lammps/src directory (not available with CPU version D3 `pair_d3.cpp`)

2. Configure `CMakeLists.txt` in the lammps/cmake directory
  - Change: `project(lammps CXX)` -> `project(lammps CXX CUDA)`
  - Change: `${LAMMPS_SOURCE_DIR}/[^.]*.cpp` -> `${LAMMPS_SOURCE_DIR}/[^.]*.cpp  ${LAMMPS_SOURCE_DIR}/[^.]*.cu`
  - Change: `set(CMAKE_CXX_STANDARD 11)` -> `set(CMAKE_CXX_STANDARD 14)`
  - Add to the last line:
    ```
    find_package(CUDA)
    target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)
  
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
    ```

3. Enter command in the lammps directory.
  ```
  mkdir build
  cd build

  cmake ../cmake -C ../cmake/presets/gcc.cmake \
  -D BUILD_MPI=no -D BUILD_OMP=no \
  -D CMAKE_CXX_FLAGS="-O3" \
  -D CMAKE_CUDA_FLAGS="-fmad=false -O3" \
  -D CMAKE_CUDA_ARCHITECTURES="86;80;70;61" \
  -D CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")

  # CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)") uses libtorch in the pytorch in your environment.
  # If you intend to use a separately installed libtorch, you can simply specify its path directly. (pre-cxx11 and cxx11 tested)

  make -j8
  ```

### Notes
- `fmad=false` is essential to obtain precise figures. Be careful to ensure that the result value is correct.
- CMAKE_CUDA_ARCHITECUTRES lists
  - 61 -> Titan X, P6000
  - 70 -> v100
  - 80 -> a100
  - 86 -> 3090ti, a5000
- If there is no GPU on the node you are compling, CMake can cause errors. (maybe)

## To do
- Implement without Unified Memory.
- Unfix the threadsPerBlock=128.
- Unroll the repetition loop k (for small number of atoms)

## Cautions
- It can be slower than the CPU with a small number of atoms.
- The CUDA math library differs from C, which can lead to numerical errors.
- The maximum number of atoms that can be calculated is 46,340. (overflow issue)