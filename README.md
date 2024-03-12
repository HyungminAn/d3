# D3 dispersion correction on LAMMPS with CUDA

Only NVIDIA GPU supported.

This is for avoiding collision between openACC and pyTorch.

The parallelization used is the same as openACC version.

# Guide
Build information
- My module lists
  - mkl/2022.1.0 mpi/2021.6.0 CUDA/12.1.0 (odin/loki server)
- Use any compiler supporting nvcc (recommend g++)
  - g++ 12.1.1 verified (odin/loki server)
- LAMMPS `23Jun2022` verified.
- Modify CMakeLists.txt below
  - `project(lammps CXX)` -> `project(lammps CXX CUDA)`
  - add `find_package(CUDA)` below project
  - `${LAMMPS_SOURCE_DIR}/[^.]*.cpp` -> `${LAMMPS_SOURCE_DIR}/*.cpp ${LAMMPS_SOURCE_DIR}/*.cu)`
  - add `target_link_libraries(lammps PUBLIC ${CUDA_LIBRARIES} cuda)` at the end

- Build LAMMPS with the command below
```
cmake ../cmake -C ../cmake/presets/gcc.cmake \
-D BUILD_MPI=no -D BUILD_OMP=no \
-D CMAKE_CXX_FLAGS="-O3" \
-D CMAKE_CUDA_FLAGS="-fmad=false -O3" \
-D CMAKE_CUDA_ARCHITECTURES="86;80;70;61" \

make -j8
```
- If you compiled with SevenNet, follow the insturctions of SevenNet + LAMMPS
- and add this flag on the cmake command
  - `-D CMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path')`

Notes
- `fmad=false` is essential to get precise figures. Be careful that the result value is correct.
- CUDA_ARCHITECUTRE
  - 61 -> Titan X, P6000
  - 70 -> v100
  - 80 -> a100
  - 86 -> 3090ti, a5000
- If there is a GPU for the node you are compiling, Cmake will find it, so CMAKE_CUDA_ARCHITECUTRES is unnecessary (maybe)
- If there is no GPU in node compilng, CMake can cause errors.


# To do
- implement without Unified memory

# Cautions
- It can be slower than the CPU with a small number of atoms.
- The CUDA math library differs from C, which can lead to numerical errors.