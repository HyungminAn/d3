# D3 dispersion correction on LAMMPS with CUDA

Only NVIDIA GPU supported.

This is for avoiding collision between openACC and pyTorch.

The parallelization used is the same as openACC version.

# Guide
Build information
- Use any compiler supporting nvcc (recommend g++)
  - ? for neuron server
  - CUDA/12.1.0 for odin/loki server
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
- If you compiled with SevenNet, follow the insturctions of SevenNet + LAMMPS and just add the flag above
  - -D CMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`

Notes
- `fmad=false` is essential to get precise figures. Be careful that the result value is correct.
- CUDA_ARCHITECUTRE
  - 61 -> Titan X, P6000
  - 70 -> v100
  - 80 -> a100
  - 86 -> 3090ti, a5000


# To do
- implement zero / zerom damping
- implement without Unified memory
