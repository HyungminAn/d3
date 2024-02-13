# D3
LAMMPS implementation of dft-d3 

(by Stefan Grimme, Jens Antony, Stephan Ehrlich, and Helge Krieg, J. Chem. Phys. 132, 154104 (2010); DOI:10.1063/1.3382344)

You can find the original version of D3 at https://www.chemie.uni-bonn.de/grimme/de/software/dft-d3.

# How to use
1. Put `pair_d3.cpp`, `pair_d3.h` into `lammps/src` directory and compile LAMMPS.
   To use OpenMP, the `-fopenmp` tag must be set.
2. Write LAMMPS input scripts like below to use D3.
```
variable        path_r0ab  string  "r0ab.csv"
variable        path_c6ab  string  "d3_pars.csv"
variable        cutoff_d3       equal   9000
variable        cutoff_d3_CN    equal   1600
variable        elem1   string  "O"
variable        elem2   string  "C"
variable        elem3   string  "H"
variable        functional_type   string   "pbe"

pair_style      d3    ${cutoff_d3}  ${cutoff_d3_CN}
pair_coeff * *  ${path_r0ab} ${path_c6ab} ${functional_type} ${elem1} ${elem2} ${elem3}
```
Here, `cutoff_d3` and `cutoff_d3_CN` are square of cutoff radii for energy/force and coordination number, respectively.

Units are Bohr radius: 1 (Bohr radius) = 0.52917721 (Å)

(Default values are 9000 and 1600, respectively)

# Features
- selective/no periodic boundary condition : not implemented yet

# Versions
1. OpenMP : current state of the art
2. MPI : in future plan
3. Using accelerators
