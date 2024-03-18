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

# To use `pair_d3` with LAMMPS `hybrid`, `hybrid/overlay`
In case you are doing calculation where pressure also affects simulation   
(ex. NPT molecular dynamics simulation, geometry optimization with cell relax...):   
***you must add `compute (name_of_your_compute) all pressure NULL virial pair/hybrid d3` to your lammps input script.***

In D3, the result of computation (energy, force, stress) will be updated to actual variables in `update` function:
```cpp
void PairD3::update(int eflag, int vflag) {
    int n = atom->natoms;
    // Energy update
    if (eflag) { eng_vdwl += disp_total * AU_TO_EV; }

    double** f_local = atom->f;       // Local force of atoms
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            f_local[i][j] += f[i][j] * AU_TO_EV / AU_TO_ANG;
        }
    }

    // Stress update
    if (vflag) {
        virial[0] += sigma[0][0] * AU_TO_EV;
        virial[1] += sigma[1][1] * AU_TO_EV;
        virial[2] += sigma[2][2] * AU_TO_EV;
        virial[3] += sigma[0][1] * AU_TO_EV;
        virial[4] += sigma[0][2] * AU_TO_EV;
        virial[5] += sigma[1][2] * AU_TO_EV;
    }
}
```
In this code, virial stresses are updated only when `vflag` turned on.  
However, the `pair_hybrid.cpp` in `lammps/src` explains how virial stresses are calculated in hybrid style:  
```cpp
/* ----------------------------------------------------------------------
  call each sub-style's compute() or compute_outer() function
  accumulate sub-style global/peratom energy/virial in hybrid
  for global vflag = VIRIAL_PAIR:
    each sub-style computes own virial[6]
    sum sub-style virial[6] to hybrid's virial[6]
  for global vflag = VIRIAL_FDOTR:
    call sub-style with adjusted vflag to prevent it calling
      virial_fdotr_compute()
    hybrid calls virial_fdotr_compute() on final accumulated f
------------------------------------------------------------------------- */
```
Here, `compute pressure` with `pair/hybrid` will switch on the `VIRIAL_PAIR` style, so the virial stress will accumulated to `hybrid/overlay`.   
Otherwise, `VIRIAL_FDOTR` may be turned on (which may skip the `vflag` part in `update` function; is it default?) and will give some errorneous value (computed from accumulated forces).   
※ The `pair_style` after `compute pressure` can be any pair_style; only the `VIRIAL_PAIR` matters in this case.

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
