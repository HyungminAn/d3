# D3 dispersion correction on LAMMPS with openACC

only NVIDIA GPU supported.

# Guide
- use pgi compiler(pgc++) for openACC (or higher version of GCC with openACC support)
  - gcc/12.2.0   cmake/3.26.2   nvidia_hpc_sdk/22.7 for neuron server
  - NV_HPC/23.3 for odin/loki server
- build LAMMPS with the command below
```
cmake ../cmake/ -C ../cmake/presets/pgi.cmake -D BUILD_MPI=no -D BUILD_OMP=no -D CMAKE_CXX_FLAGS="-acc=gpu -gpu=managed -Minfo=accel -fast" -D CMAKE_C_FLAGS="-acc=gpu -gpu=managed -Minfo=accel -fast"
make -j 4
```

# To do
- implement zero / zerom damping
- compile with SimpleNN / SevenNet 
- implement without gpu=managed (without shared memory)
- optimize for specific GPU architecture
