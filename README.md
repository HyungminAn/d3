# D3 dispersion on LAMMPS with openACC

only NVIDIA GPU supported.

# Guide
- gcc/12.2.0   cmake/3.26.2   nvidia_hpc_sdk/22.7 for neuron server
- NV_HPC/23.3 for odin/loki server
- requirements: pgi compiler(pgc++) for openACC (or higher version of GCC with openACC support)
- LAMMPS build with the command below

```
cmake ../cmake/ -C ../cmake/presets/pgi.cmake -D BUILD_MPI=no -D BUILD_OMP=no -D CMAKE_CXX_FLAGS="-acc=gpu -gpu=managed -Minfo=accel -fast" -D CMAKE_C_FLAGS="-acc=gpu -gpu=managed -Minfo=accel -fast"
```

# To do
- implement without gpu=managed (without shared memory)
- optimization for GPU architecture