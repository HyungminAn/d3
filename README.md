# D3
LAMMPS implementation of [D3](https://doi.org/10.1063/1.3382344).   

You can find the original FORTRAN code of [dftd3](https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3).

# How to use
1. Put `pair_d3.cpp`, `pair_d3.h` into `lammps/src` directory and compile LAMMPS.   
   To use OpenMP, the `-fopenmp` tag must be set.

2. Write LAMMPS input scripts like below to use D3.
```vim
variable        path_r0ab  string  "r0ab.csv"
variable        path_c6ab  string  "d3_pars.csv"
variable        cutoff_d3       equal   9000
variable        cutoff_d3_CN    equal   1600
variable        damping_type    string  "d3_damp_bj"
variable        functional_type   string   "pbe"
variable        elem_list       string  "O C H"

pair_style      d3    ${cutoff_d3}  ${cutoff_d3_CN}  ${damping_type}
pair_coeff * *  ${path_r0ab} ${path_c6ab} ${functional_type} ${elem_list}
```

`r0ab.csv` and `d3_pars.csv` files should exist to calculate d3 interactions (those files are in `lammps_test` folder).

`cutoff_d3` and `cutoff_d3_CN` are *square* of cutoff radii for energy/force and coordination number, respectively.   
Units are Bohr radius: 1 (Bohr radius) = 0.52917721 (Å)   
(Default values are 9000 and 1600, respectively)

Available damping types: `d3_damp_zero`, `d3_damp_bj`, `d3_damp_zerom`, `d3_damp_bjm`   
(Zero damping, Becke-Johnson damping and their modified versions, respectively)

3. Run your LAMMPS code.

To use OpenMP version, you should set `OMP_NUM_THREADS` variable adequately to make full use of your CPU cores.
```bash
export OMP_NUM_THREADS=32
lmp -in lammps.in
```
or
```bash
env OMP_NUM_THREADS=32 lmp -in lammps.in
```

# Note
1. In [VASP DFT-D3](https://www.vasp.at/wiki/index.php/DFT-D3) page, `VDW_RADIUS` and `VDW_CNRADIUS` are `50.2` and `20.0`, respectively (units are Å).   
But you can check the default value of these in OUTCAR: `50.2022` and `21.1671`, which is same to default values of this code.   
To check this by yourself, run VASP with D3 using zero damping (BJ does not give such log).

# Features
- selective/no periodic boundary condition : implemented   
  (But only PBC/noPBC can be checked through original FORTRAN code; selective PBC cannot)
- 3-body term, n > 8 term : not implemented   
  (Same condition to VASP)

# Versions
1. OpenMP : current state of the art
2. MPI : in future plan
3. Using accelerators
