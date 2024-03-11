# D3 dispersion correction on LAMMPS with CUDA

Only NVIDIA GPU supported.

This is for avoiding collision between openACC and pyTorch.

The parallelization used is the same as openACC version.

# Guide
- Use any compiler supporting nvcc (recommend g++)
  - ? for neuron server
  - CUDA/12.1.0 for odin/loki server
- LAMMPS `23Jun2022` verified.
- Build LAMMPS with the command below
```
cmake ../cmake -C ../cmake/presets/gcc.cmake \
-D BUILD_MPI=no -D BUILD_OMP=no \
-D CMAKE_CXX_FLAGS="-O3" \
-D CMAKE_CUDA_FLAGS="-arch=sm_86 -fmad=false -O3" \
-D CMAKE_CUDA_ARCHITECTURES=86

make -j8
```

`fmad=false` is essential to get precise figures.

`sm_86` is optimal for a5000 and 3090ti

Be careful that the result value is correct.



# To do
- implement zero / zerom damping
- implement without Unified memory
